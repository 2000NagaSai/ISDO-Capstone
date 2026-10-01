"""
ISDO Lab C6/C7 — LangGraph Orchestrator: wires the ISDO agents into one StateGraph.

    triage -> resolution -> sla --(hitl_required)--> hitl -> communication -> END
                                 \\------(otherwise)-------> communication -> END

C7: the HITL gate fires for any of three reasons (several can apply at once):
    1. P1 ticket with SLA CRITICAL/BREACHED
    2. Resolution Agent confidence is LOW (any priority)
    3. Access Grant request (request_type 'Access Grant') — any priority

C8: when the local ChromaDB search gives LOW confidence, the Resolution node asks the
A2A Knowledge Specialist (a2a/knowledge_specialist.py, port 8001) for a second opinion.
If it returns MEDIUM/HIGH, its resolution and confidence replace the local ones (so the
LOW-confidence HITL trigger no longer fires). If it is not running or fails, the ticket
stays LOW and goes to the HITL gate.

C9: PII guardrail + persistent audit trail.
    - triage_node redacts short_description + description (guardrails/pii_redactor.py) and stores
      the redacted text + token mapping in TicketState. EVERY later Claude call (triage, resolution,
      A2A, communication) uses only the redacted text.
    - communication_node: Claude drafts with tokens ([NAME_1]...), restore() puts the real values
      back, and only then is the message posted to the ServiceNow / Jira mock.
    - One AuditLogger writes each audit entry to logs/audit_trail.jsonl as it happens.
    - A PII leak check records every request sent to Claude and reports any original PII value
      that appears in it.

Reuses the agents from Labs C3-C5 (agents/triage_agent.py, resolution_agent.py, sla_agent.py).

Run from the project root:
    python orchestrator/supervisor.py                 # all test tickets
    python orchestrator/supervisor.py INC0001002      # only the listed ticket(s)
Needs: ANTHROPIC_API_KEY in .env, Lab C1 KB (data/chroma_db), snow_shim + jira_shim running (Lab C2).
Optional (C8): the Knowledge Specialist on port 8001.
"""

import json
import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import requests
from langgraph.graph import END, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agents"))
sys.path.insert(0, str(PROJECT_ROOT))                    # for guardrails.pii_redactor

import resolution_agent  # noqa: E402  (C4 — connects to the ChromaDB KB on import)
import sla_agent         # noqa: E402  (C5)
import triage_agent      # noqa: E402  (C3)
from guardrails.pii_redactor import AuditLogger, redact, restore  # noqa: E402  (C9)

try:
    sys.stdout.reconfigure(encoding="utf-8")   # so ▶ / ✅ print on any Windows console
except Exception:
    pass

client, MODEL = triage_agent.client, triage_agent.MODEL
PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
JIRA_ISSUE_URL = "http://localhost:5002/rest/api/2/issue"
ACCESS_GRANT = "access grant"
A2A_URL = os.environ.get("ISDO_A2A_URL", "http://localhost:8001")   # Knowledge Specialist (C8)
A2A_TIMEOUT = 90          # seconds — POST /tasks runs a Claude call before it answers
CONFIDENCE_LEVELS = {"HIGH", "MEDIUM", "LOW"}
SNOW_INCIDENT_URL = "http://localhost:5001/api/now/table/incident"

# C9: ONE audit logger for the whole run — every node's audit entry is written to this file as it happens
AUDIT_TRAIL_FILE = PROJECT_ROOT / "logs" / "audit_trail.jsonl"
AUDIT_LOGGER = AuditLogger(str(AUDIT_TRAIL_FILE))
FIELD_SEP = "\n<<<ISDO-FIELD-SEPARATOR>>>\n"   # lets both fields share one redaction mapping

# ── C9: record every request sent to Claude, to prove PII never reaches it ────
CLAUDE_INPUTS = []      # [{"ticket": ..., "caller": ..., "payload": "<json text sent to Claude>"}]
_current_ticket = {"number": ""}


def _watch_claude(api_client, caller):
    """Wrap client.messages.create so every outgoing request is recorded (system + messages + tools)."""
    original_create = api_client.messages.create

    def create(**kwargs):
        payload = {k: kwargs.get(k) for k in ("system", "messages")}
        CLAUDE_INPUTS.append({"ticket": _current_ticket["number"], "caller": caller,
                              "payload": json.dumps(payload, default=str)})
        return original_create(**kwargs)

    api_client.messages.create = create


_watch_claude(triage_agent.client, "TriageAgent / CommunicationAgent")
if resolution_agent.client is not triage_agent.client:
    _watch_claude(resolution_agent.client, "ResolutionAgent")

# ── SHARED STATE ──────────────────────────────────────────────────────────────
class TicketState(TypedDict, total=False):
    # input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str          # Jira request type for REQ- tickets, e.g. "Access Grant"
    # C9 guardrail — set by triage_node; only the redacted fields are ever sent to Claude
    redacted_short_description: str
    redacted_description: str
    pii_mapping: dict          # {"[NAME_1]": "John Smith", ...} — never logged, only used by restore()
    # triage
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    # resolution
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    # sla / hitl
    sla_breach_risk: str
    escalation_required: bool
    hitl_required: bool
    hitl_reason: str           # C7: why the gate fired (reasons joined with " | ")
    a2a_used: bool             # C8: resolution/confidence came from the Knowledge Specialist
    a2a_status: str            # C8: not_needed / completed / unavailable / error: ...
    hitl_approved: bool
    # communication
    user_message: str
    final_status: str
    # every node appends; the reducer concatenates instead of overwriting
    audit_log: Annotated[list, operator.add]


def audit(state, agent, action, detail, approval="Auto"):
    """One audit entry: written to logs/audit_trail.jsonl immediately (C9) and returned for
    TicketState['audit_log'] (the reducer appends it). AuditLogger prints the [AUDIT] line."""
    AUDIT_LOGGER.log(agent, action, state.get("ticket_number", ""), tool=action,
                     rationale=detail, approval_status=approval)
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail, "approval_status": approval}]


def effective_priority(state):
    """The more severe of the ticket's recorded priority and the triage priority."""
    candidates = [p for p in (state.get("priority"), state.get("triage_priority")) if p in PRIORITY_RANK]
    return min(candidates, key=PRIORITY_RANK.get) if candidates else "P3"


def escalation_team(state):
    return sla_agent.ESCALATION_TEAMS.get(state.get("triage_category"), sla_agent.DEFAULT_TEAM)


def lookup_request_type(ticket_number):
    """For REQ- tickets without a request_type, ask the Jira shim (Lab C2). None if unavailable."""
    if not ticket_number.startswith("REQ-"):
        return None
    try:
        r = requests.get(f"{JIRA_ISSUE_URL}/{ticket_number}", timeout=3)
        return r.json()["fields"]["issuetype"]["name"] if r.ok else None
    except (requests.exceptions.RequestException, KeyError, ValueError):
        return None


def is_access_grant(state):
    # Deliberately keyed on request_type alone: an access grant must never skip the gate
    # just because triage put it in a different category.
    return (state.get("request_type") or "").strip().lower() == ACCESS_GRANT


def update_jira_request(ticket_number, status, note):
    """Set a Jira request's status on the Jira shim (Lab C2); simulated if the shim is down."""
    try:
        r = requests.put(f"{JIRA_ISSUE_URL}/{ticket_number}",
                         json={"fields": {"status": {"name": status}, "work_notes": note}}, timeout=5)
        ok, source = r.ok, "jira_shim"
    except requests.exceptions.ConnectionError:
        ok, source = True, "simulated (jira_shim not running)"
    print(f"  [Jira Mock] {ticket_number} status -> {status}" + ("" if ok else "  FAILED"))
    return ok


def call_knowledge_specialist(state):
    """A2A call (Lab C8): POST /tasks -> task_id, then GET /tasks/{task_id} -> result.
    Returns the task's 'result' dict. Raises requests exceptions / ValueError on any failure."""
    payload = {   # C9: redacted text only — the specialist sends this to Claude too
        "query": f"{state['redacted_short_description']}. {state['redacted_description']}",
        "ticket_number": state["ticket_number"],
        "context": f"category={state.get('triage_category')}, priority={effective_priority(state)}",
    }
    print(f"  -> A2A: POST {A2A_URL}/tasks")
    CLAUDE_INPUTS.append({"ticket": state["ticket_number"], "caller": "A2A Knowledge Specialist",
                          "payload": json.dumps(payload)})
    r = requests.post(f"{A2A_URL}/tasks", json=payload, timeout=A2A_TIMEOUT)
    r.raise_for_status()
    task_id = r.json()["task_id"]

    print(f"  -> A2A: GET  {A2A_URL}/tasks/{task_id}")
    r = requests.get(f"{A2A_URL}/tasks/{task_id}", timeout=A2A_TIMEOUT)
    r.raise_for_status()
    task = r.json()
    if task.get("status") != "completed":
        raise ValueError(f"task {task_id} status is {task.get('status')!r}, not 'completed'")
    return task["result"]

# ── NODES ─────────────────────────────────────────────────────────────────────
def triage_node(state: TicketState) -> dict:
    print(f"\n▶ TRIAGE AGENT — {state['ticket_number']}")

    # ── C9: redact BEFORE anything goes to Claude. Both fields are redacted in one call so the
    #    same person/email gets the same token in both (separate calls would each start at _1).
    combined, mapping = redact(state["short_description"] + FIELD_SEP + state["description"])
    clean_short, _, clean_desc = combined.partition(FIELD_SEP)
    entries = audit(state, "PIIGuardrail", "redact_pii",
                    f"{len(mapping)} PII item(s) masked before Claude: {sorted(mapping) or 'none'}")
    if mapping:
        print(f"  PII masked: {sorted(mapping)}")
        print(f"  Claude sees: {clean_short} | {clean_desc}")

    c = triage_agent.triage_ticket(state["ticket_number"], clean_short, clean_desc)
    if c is None:   # agent failed to classify: fall back to the ticket's own fields
        c = {"category": state.get("category", "Unknown"), "priority": state.get("priority", "P3"),
             "assignment_group": "Service-Desk", "pii_detected": False, "reasoning": "triage failed - fallback"}
    # Claude only sees tokens (and its prompt says placeholders are not PII), so the redactor's
    # own finding counts too.
    pii = bool(c["pii_detected"]) or bool(mapping)
    entries += audit(state, "TriageAgent", "classify_ticket",
                     f"{c['category']} / {c['priority']} -> {c['assignment_group']}; "
                     f"PII={pii}; {c.get('reasoning', '')}")
    return {
        "redacted_short_description": clean_short,
        "redacted_description": clean_desc,
        "pii_mapping": mapping,
        "triage_category": c["category"],
        "triage_priority": c["priority"],
        "triage_assignment_group": c["assignment_group"],
        "pii_detected": pii,
        "audit_log": entries,
    }


def resolution_node(state: TicketState) -> dict:
    print("\n▶ RESOLUTION AGENT — searching KB")
    r = resolution_agent.resolve_ticket(       # C9: redacted text only
        state["ticket_number"], state["redacted_short_description"], state["redacted_description"],
        state.get("triage_category", state.get("category")), effective_priority(state))
    update = {
        "kb_article": r.get("kb_article_used", "none"),
        "resolution_text": r.get("resolution_text", ""),
        "auto_resolve": bool(r.get("auto_resolve")),
        "confidence": r.get("confidence", "LOW"),
        "a2a_used": False,
        "a2a_status": "not_needed",
        "audit_log": audit(state, "ResolutionAgent", "search_kb",
                           f"{r.get('kb_article_used')} | {r.get('confidence')} ({r.get('score', 0):.0%}) | "
                           f"auto_resolve={r.get('auto_resolve')}"),
    }
    if update["confidence"] != "LOW":
        return update

    # ── C8: LOW confidence -> ask the Knowledge Specialist over A2A ──
    print("\n▶ A2A — LOW confidence, asking Knowledge Specialist")
    try:
        res = call_knowledge_specialist(state)
    except requests.exceptions.ConnectionError:
        print(f"  A2A server not reachable at {A2A_URL} — falling back to HITL")
        update["a2a_status"] = "unavailable"
        update["audit_log"] += audit(state, "ResolutionAgent", "a2a_call", f"FAILED: {A2A_URL} not reachable -> HITL")
        return update
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        print(f"  A2A call failed ({e}) — falling back to HITL")
        update["a2a_status"] = f"error: {type(e).__name__}"
        update["audit_log"] += audit(state, "ResolutionAgent", "a2a_call", f"FAILED: {e} -> HITL")
        return update

    conf = str(res.get("confidence", "LOW")).upper()
    if conf not in CONFIDENCE_LEVELS or res.get("escalate_to_l2"):
        conf = "LOW"               # unknown value or specialist says escalate -> treat as LOW
    print(f"  A2A result: {res.get('best_match')} | {conf} ({res.get('confidence_score', 0):.0%}) | "
          f"escalate_to_l2={res.get('escalate_to_l2')}")
    update.update({
        "a2a_used": True,
        "a2a_status": "completed",
        "confidence": conf,
        "kb_article": f"{res.get('best_match', 'none')} (via A2A)",
        "resolution_text": res.get("resolution") or update["resolution_text"],
        # The specialist writes for L2 engineers, so its answer is never auto-sent to the user.
        "auto_resolve": False,
    })
    update["audit_log"] += audit(state, "ResolutionAgent", "a2a_call",
                                 f"Knowledge Specialist: {res.get('best_match')} | {conf} "
                                 f"({res.get('confidence_score', 0):.0%}) | escalate_to_l2={res.get('escalate_to_l2')}")
    return update


def sla_node(state: TicketState) -> dict:
    print("\n▶ SLA AGENT — checking deadline")
    priority = effective_priority(state)
    s = sla_agent.get_sla_status(state["ticket_number"], state["sla_due"], priority)
    if "error" in s:
        print(f"  SLA check failed: {s['error']}")
        s = {"breach_risk": "UNKNOWN", "minutes_remaining": None, "requires_escalation": False}
    print(f"  SLA Risk: {s['breach_risk']}  |  Minutes remaining: {s['minutes_remaining']}")

    escalation_required = bool(s["requires_escalation"])

    # ── C7: collect every HITL trigger that applies ──
    reasons = []
    if escalation_required and priority in sla_agent.HITL_PRIORITIES:
        reasons.append(f"{priority} SLA {s['breach_risk']} -- escalation to {escalation_team(state)} "
                       f"requires approval")
    if state.get("confidence", "LOW") == "LOW":
        a2a = state.get("a2a_status", "not_needed")
        a2a_note = ("Knowledge Specialist also LOW" if a2a == "completed"
                    else f"Knowledge Specialist {a2a}" if a2a != "not_needed" else "")
        reasons.append(f"LOW KB confidence -- no reliable fix found (KB: {state.get('kb_article', 'none')}"
                       + (f"; {a2a_note}" if a2a_note else "")
                       + f"); route to {escalation_team(state)} for L2 handling")
    if is_access_grant(state):
        reasons.append("ACCESS GRANT -- access request requires security approval")
    hitl_required = bool(reasons)
    hitl_reason = " | ".join(reasons)

    entries = audit(state, "SLAAgent", "get_sla_status",
                    f"{priority} {s['breach_risk']}, {s['minutes_remaining']} min left; "
                    f"escalation_required={escalation_required}, hitl_required={hitl_required}"
                    + (f"; reason: {hitl_reason}" if hitl_required else ""))
    if hitl_required:
        print(f"  HITL required: {hitl_reason}")

    # Escalations that need no human (e.g. breached P2) go ahead now; gated ones wait for the HITL node
    if escalation_required and not hitl_required:
        team = escalation_team(state)
        sla_agent.update_ticket(state["ticket_number"], "escalate", team,
                                note=f"Auto-escalated: SLA {s['breach_risk']}")
        entries += audit(state, "SLAAgent", "update_ticket", f"auto-escalated to {team}")

    return {"sla_breach_risk": s["breach_risk"], "escalation_required": escalation_required,
            "hitl_required": hitl_required, "hitl_reason": hitl_reason,
            # nothing that needs a human may auto-resolve
            "auto_resolve": state.get("auto_resolve", False) and not hitl_required,
            "audit_log": entries}


def hitl_node(state: TicketState) -> dict:
    print("\n▶ HITL GATE — human approval required")
    ticket, reason = state["ticket_number"], state.get("hitl_reason", "")
    access = is_access_grant(state)
    team = escalation_team(state)
    action = "Approve access grant" if access else f"Escalate to {team}"

    print("  " + "WARNING  " * 8)
    print(f"  Ticket:  {ticket}  |  Priority: {effective_priority(state)}")
    for i, r in enumerate(reason.split(" | ")):
        print(f"  {'Reason: ' if i == 0 else '         '} {r}")
    print(f"  Action:  {action}")
    print("  " + "WARNING  " * 8)
    try:
        approved = input("  Approve action? [y/n]: ").strip().lower() == "y"
    except EOFError:                     # no terminal attached -> never approve silently
        approved = False
    decision = "APPROVED" if approved else "REJECTED"

    sla_agent.audit({"ticket": ticket, "action": action, "detail": reason, "decision": decision})  # logs/hitl_audit.jsonl
    entries = audit(state, "HITLGate", "approval_decision", f"{decision}: {action} | reason: {reason}",
                    approval=decision)
    print(f"  Decision: {decision}")

    if access:
        if approved:
            update_jira_request(ticket, "Approved", "Access grant approved by human approver")
            entries += audit(state, "HITLGate", "update_request", "Jira status -> Approved")
        else:
            update_jira_request(ticket, "Pending Approval", "Access grant not approved; awaiting review")
            entries += audit(state, "HITLGate", "update_request", "Jira status -> Pending Approval")
    elif approved:
        sla_agent.update_ticket(ticket, "escalate", team, note=f"Approved by human operator: {reason}")
        entries += audit(state, "SLAAgent", "update_ticket", f"escalated to {team}")
    else:
        sla_agent.update_ticket(ticket, "add_note", note=f"Action declined by human operator: {reason}")
        entries += audit(state, "SLAAgent", "update_ticket", "added note: action declined")
    return {"hitl_approved": approved, "audit_log": entries}


def post_user_message(ticket_number, message):
    """C9: post the (restored) user message to the system of record — ServiceNow mock for INC,
    Jira mock for REQ. Returns a short description of what happened."""
    try:
        if ticket_number.startswith("REQ-"):
            r = requests.put(f"{JIRA_ISSUE_URL}/{ticket_number}",
                             json={"fields": {"comment": message}}, timeout=5)
            target = "Jira mock"
        else:
            r = requests.patch(f"{SNOW_INCIDENT_URL}/{ticket_number}", json={"comments": message}, timeout=5)
            target = "ServiceNow mock"
        return f"posted to {target}" if r.ok else f"{target} rejected it (HTTP {r.status_code})"
    except requests.exceptions.ConnectionError:
        return "not posted (mock API not running)"


def communication_node(state: TicketState) -> dict:
    print("\n▶ COMMUNICATION AGENT")
    access = is_access_grant(state)
    mapping = state.get("pii_mapping") or {}
    name_tokens = [t for t in mapping if t.startswith("[NAME_")]
    if name_tokens:                       # Claude greets the person by TOKEN; restore() fills the name in
        greeting = f"Dear {name_tokens[0]},"
    else:
        greeting = "Dear Requester," if state["ticket_number"].startswith("REQ-") else "Dear User,"

    if state.get("auto_resolve"):
        kind, status = "self-service resolution with the steps below", "RESOLVED"
    elif state.get("hitl_required") and not state.get("hitl_approved"):
        kind, status = ("'pending approval' notice: the request is awaiting approval and the requester "
                        "will be updated once a decision is made"), "PENDING_APPROVAL"
    elif state.get("hitl_approved") and access:
        kind, status = ("access grant approval notice: the access request has been approved and will be "
                        "fulfilled by the responsible team"), "APPROVED"
    elif state.get("hitl_approved") or state.get("escalation_required"):
        kind, status = "escalation confirmation (a senior team has been engaged)", "ESCALATED"
    else:
        kind, status = "assignment notification (the ticket is with the support team)", "ASSIGNED"

    # C9: Claude only ever sees the redacted fields
    facts = (f"Ticket: {state['ticket_number']}\nIssue: {state['redacted_short_description']}\n"
             f"Details: {state['redacted_description']}\n"
             f"Assigned team: {state.get('triage_assignment_group')}\nPriority: {effective_priority(state)}\n")
    if status == "RESOLVED":
        facts += f"Resolution steps:\n{state.get('resolution_text', '')}\n"

    try:
        response = client.messages.create(
            model=MODEL, max_tokens=600,
            system=f"You write short, polite IT service desk messages. Plain text, no markdown headings, "
                   f"under 120 words. Start with '{greeting}'. Placeholders in square brackets such as "
                   f"[NAME_1] or [EMAIL_1] stand for personal data: copy them exactly as written if you need "
                   f"them, never guess or invent the real values, and never write any other names, emails "
                   f"or IDs. Do not promise times that are not given. Do not mention internal approval reasons.",
            messages=[{"role": "user", "content": f"Write a {kind} for this ticket.\n\n{facts}"}],
        )
        draft = "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as e:           # never lose the ticket because the message draft failed
        print(f"  (message draft failed: {e}; using template)")
        draft = f"{greeting} regarding {state['ticket_number']} ({state['redacted_short_description']}): {kind}."

    # C9: restore() only now — after Claude, right before the message leaves for ServiceNow / Jira
    message = restore(draft, mapping)
    posted = post_user_message(state["ticket_number"], message)

    print("  DRAFT FROM CLAUDE (masked):\n  " + draft.replace("\n", "\n  "))
    if mapping:
        print("  SENT TO USER (restored):\n  " + message.replace("\n", "\n  "))
    print(f"  -> {posted}")
    print(f"\n✅ FINAL STATUS: {status}")
    entries = audit(state, "CommunicationAgent", "draft_user_message", f"final_status={status}")
    entries += audit(state, "CommunicationAgent", "post_comment",
                     f"user message {posted}; {len(mapping)} PII token(s) restored after Claude")
    return {"user_message": message, "final_status": status, "audit_log": entries}

# ── GRAPH ─────────────────────────────────────────────────────────────────────
def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("triage", triage_node)
    g.add_node("resolution", resolution_node)
    g.add_node("sla", sla_node)
    g.add_node("hitl", hitl_node)
    g.add_node("communication", communication_node)

    g.set_entry_point("triage")
    g.add_edge("triage", "resolution")
    g.add_edge("resolution", "sla")
    g.add_conditional_edges("sla", route_after_sla, {"hitl": "hitl", "communication": "communication"})
    g.add_edge("hitl", "communication")
    g.add_edge("communication", END)
    return g.compile()


graph = build_graph()


def process_ticket(ticket: dict) -> dict:
    print(f"\n{'═' * 55}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'═' * 55}")
    _current_ticket["number"] = ticket["ticket_number"]
    ticket = dict(ticket)
    if not ticket.get("request_type"):
        ticket["request_type"] = lookup_request_type(ticket["ticket_number"]) or ""
    return graph.invoke({**ticket, "audit_log": []})

# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated now = 2024-01-15 10:30 (Lab C5). Same demo SLA times as Lab C5.
    test_tickets = [
        {   # P2, 90 min left (AT_RISK), HIGH confidence -> no HITL -> RESOLVED
            "ticket_number": "INC0001001",
            "short_description": "VPN not connecting after password change",
            "description": "User reports VPN client fails to connect after AD password was reset. "
                           "Error: authentication failed.",
            "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00",
        },
        # Step 3 — LOW-confidence trigger: comment out the VPN ticket above and uncomment this one.
        # (Change the description too — with the VPN description the KB still finds the VPN article.)
        # {   # P3, no KB article covers Webex -> LOW confidence -> HITL even at P3
        #     "ticket_number": "INC0001001",
        #     "short_description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
        #     "description": "Cisco Webex app crashes on launch on a MacBook M2 since the macOS Sonoma update.",
        #     "category": "Software", "priority": "P3", "sla_due": "2024-01-15 16:00:00",
        # },
        {   # P1, 10 min left (CRITICAL) -> HITL (Step 2: run once with y, once with n)
            "ticket_number": "INC0001002",
            "short_description": "Cannot access ERP system - login error",
            "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
                           "Started 09:00 today.",
            "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00",
        },
        {   # C8 — no KB article covers BSOD -> LOW locally -> A2A Knowledge Specialist
            #      (A2A running: its confidence is used; A2A down: falls back to HITL)
            "ticket_number": "INC0001012",
            "short_description": "Blue screen error on workstation",
            "description": "User workstation showing BSOD with error SYSTEM_SERVICE_EXCEPTION. "
                           "Happens 2-3 times per day.",
            "category": "Hardware", "priority": "P2", "sla_due": "2024-01-16 16:00:00",
        },
        {   # C9 — PII in the ticket: name, employee ID, email, phone. Claude must only see tokens.
            #      P2, password reset KB article -> auto-resolve; the reply is restored to "Dear Priya Sharma"
            "ticket_number": "INC0001006",
            "short_description": "Password reset request for Priya Sharma",
            "description": "User Priya Sharma (emp ID ZEN-4471) locked out of AD account after 5 failed "
                           "attempts. Contact: priya.sharma@zensar.com or +91-9876543210.",
            "category": "Access", "priority": "P2", "sla_due": "2024-01-15 16:00:00",
        },
        {   # Step 4 — Access Grant, P2 -> HITL regardless of priority
            "ticket_number": "REQ-1002",
            "short_description": "VPN access for new contractor",
            "description": "Contractor needs VPN access. Email: contractor@client.com",
            "category": "Access", "priority": "P2", "sla_due": "2024-01-15 15:00:00",
            "request_type": "Access Grant",   # if omitted, looked up from the Jira shim
        },
    ]

    selected = set(sys.argv[1:])
    if selected:
        test_tickets = [t for t in test_tickets if t["ticket_number"] in selected]
        if not test_tickets:
            sys.exit(f"No test ticket matches {sorted(selected)}")

    results = [process_ticket(t) for t in test_tickets]

    # ── 1. Audit trail per ticket, with final status ──
    for r in results:
        path = " → ".join(dict.fromkeys(e["agent"].replace("Agent", "").replace("Gate", "")
                                        .replace("PIIGuardrail", "PII") for e in r["audit_log"]))
        print(f"\n{'═' * 55}\nAUDIT LOG: {r['ticket_number']}  |  FINAL STATUS: {r['final_status']}  |  {path}"
              f"\n{'═' * 55}")
        for e in r["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19} {e['action']:<19} {e['approval_status']:<9} {e['detail']}")

    # ── 2. What Claude saw: original vs redacted ──
    print(f"\n{'═' * 55}\nPII GUARDRAIL — WHAT CLAUDE SAW\n{'═' * 55}")
    for r in results:
        if not r.get("pii_mapping"):
            print(f"  {r['ticket_number']}: no PII found")
            continue
        print(f"  {r['ticket_number']}: {len(r['pii_mapping'])} item(s) masked -> {sorted(r['pii_mapping'])}")
        print(f"     original : {r['description']}")
        print(f"     to Claude: {r['redacted_description']}")

    # ── 3. Leak check: did any PII value appear in ANY request sent to Claude? ──
    # Checks every value the redactor detected PLUS the PII we know we planted in the test tickets,
    # so a value the redactor MISSED is reported as a leak instead of passing silently.
    KNOWN_TEST_PII = {
        "INC0001006": ["Priya Sharma", "ZEN-4471", "priya.sharma@zensar.com", "9876543210"],
        "REQ-1002": ["contractor@client.com"],
    }
    values = {(r["ticket_number"], v) for r in results for v in (r.get("pii_mapping") or {}).values()}
    values |= {(t, v) for t, vs in KNOWN_TEST_PII.items() if t in {r["ticket_number"] for r in results}
               for v in vs}
    leaks = sorted({(c["ticket"], c["caller"], v) for c in CLAUDE_INPUTS for t, v in values
                    if t == c["ticket"] and v.lower() in c["payload"].lower()})
    print(f"\n{'═' * 55}\nPII LEAK CHECK — {len(CLAUDE_INPUTS)} request(s) sent to Claude / A2A\n{'═' * 55}")
    for ticket, caller, value in leaks:
        print(f"  ❌ LEAK: {ticket} — {caller} received '{value}'")
    if leaks:
        print("  -> Use the updated guardrails/pii_redactor.py and/or install the spaCy model "
              "(python -m spacy download en_core_web_sm).")
    else:
        print(f"  ✅ None of the {len(values)} PII value(s) appeared in any request to Claude.")

    print(f"\nFull audit trail ({len(AUDIT_LOGGER.entries)} entries) written to: {AUDIT_TRAIL_FILE}")
