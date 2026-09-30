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

Reuses the agents from Labs C3-C5 (agents/triage_agent.py, resolution_agent.py, sla_agent.py).

Run from the project root:
    python orchestrator/supervisor.py                 # all test tickets
    python orchestrator/supervisor.py INC0001002      # only the listed ticket(s)
Needs: ANTHROPIC_API_KEY in .env, Lab C1 KB (data/chroma_db), snow_shim + jira_shim running (Lab C2).
Optional (C8): the Knowledge Specialist on port 8001.
"""

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

import resolution_agent  # noqa: E402  (C4 — connects to the ChromaDB KB on import)
import sla_agent         # noqa: E402  (C5)
import triage_agent      # noqa: E402  (C3)

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


def audit(agent, action, detail):
    """One audit entry (Lab C9 persists these). Printed as it is written."""
    print(f"  [AUDIT] {agent}: {action}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


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
    payload = {
        "query": f"{state['short_description']}. {state['description']}",
        "ticket_number": state["ticket_number"],
        "context": f"category={state.get('triage_category')}, priority={effective_priority(state)}",
    }
    print(f"  -> A2A: POST {A2A_URL}/tasks")
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
    c = triage_agent.triage_ticket(state["ticket_number"], state["short_description"], state["description"])
    if c is None:   # agent failed to classify: fall back to the ticket's own fields
        c = {"category": state.get("category", "Unknown"), "priority": state.get("priority", "P3"),
             "assignment_group": "Service-Desk", "pii_detected": False, "reasoning": "triage failed - fallback"}
    return {
        "triage_category": c["category"],
        "triage_priority": c["priority"],
        "triage_assignment_group": c["assignment_group"],
        "pii_detected": bool(c["pii_detected"]),
        "audit_log": audit("TriageAgent", "classify_ticket",
                           f"{c['category']} / {c['priority']} -> {c['assignment_group']}; "
                           f"PII={c['pii_detected']}; {c.get('reasoning', '')}"),
    }


def resolution_node(state: TicketState) -> dict:
    print("\n▶ RESOLUTION AGENT — searching KB")
    r = resolution_agent.resolve_ticket(
        state["ticket_number"], state["short_description"], state["description"],
        state.get("triage_category", state.get("category")), effective_priority(state))
    update = {
        "kb_article": r.get("kb_article_used", "none"),
        "resolution_text": r.get("resolution_text", ""),
        "auto_resolve": bool(r.get("auto_resolve")),
        "confidence": r.get("confidence", "LOW"),
        "a2a_used": False,
        "a2a_status": "not_needed",
        "audit_log": audit("ResolutionAgent", "search_kb",
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
        update["audit_log"] += audit("ResolutionAgent", "a2a_call", f"FAILED: {A2A_URL} not reachable -> HITL")
        return update
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        print(f"  A2A call failed ({e}) — falling back to HITL")
        update["a2a_status"] = f"error: {type(e).__name__}"
        update["audit_log"] += audit("ResolutionAgent", "a2a_call", f"FAILED: {e} -> HITL")
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
    update["audit_log"] += audit("ResolutionAgent", "a2a_call",
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

    entries = audit("SLAAgent", "get_sla_status",
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
        entries += audit("SLAAgent", "update_ticket", f"auto-escalated to {team}")

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
    entries = audit("HITLGate", "approval_decision", f"{decision}: {action} | reason: {reason}")
    print(f"  Decision: {decision}")

    if access:
        if approved:
            update_jira_request(ticket, "Approved", "Access grant approved by human approver")
            entries += audit("HITLGate", "update_request", "Jira status -> Approved")
        else:
            update_jira_request(ticket, "Pending Approval", "Access grant not approved; awaiting review")
            entries += audit("HITLGate", "update_request", "Jira status -> Pending Approval")
    elif approved:
        sla_agent.update_ticket(ticket, "escalate", team, note=f"Approved by human operator: {reason}")
        entries += audit("SLAAgent", "update_ticket", f"escalated to {team}")
    else:
        sla_agent.update_ticket(ticket, "add_note", note=f"Action declined by human operator: {reason}")
        entries += audit("SLAAgent", "update_ticket", "added note: action declined")
    return {"hitl_approved": approved, "audit_log": entries}


def communication_node(state: TicketState) -> dict:
    print("\n▶ COMMUNICATION AGENT")
    access = is_access_grant(state)
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

    facts = (f"Ticket: {state['ticket_number']}\nIssue: {state['short_description']}\n"
             f"Assigned team: {state.get('triage_assignment_group')}\nPriority: {effective_priority(state)}\n")
    if status == "RESOLVED":
        facts += f"Resolution steps:\n{state.get('resolution_text', '')}\n"

    try:
        response = client.messages.create(
            model=MODEL, max_tokens=600,
            system=f"You write short, polite IT service desk messages. Plain text, no markdown headings, "
                   f"under 120 words. Start with '{greeting}'. Never include names, emails or IDs of people. "
                   f"Do not promise times that are not given. Do not mention internal approval reasons.",
            messages=[{"role": "user", "content": f"Write a {kind} for this ticket.\n\n{facts}"}],
        )
        message = "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as e:           # never lose the ticket because the message draft failed
        print(f"  (message draft failed: {e}; using template)")
        message = f"{greeting} regarding {state['ticket_number']} ({state['short_description']}): {kind}."

    print(f"  USER MESSAGE:\n  " + message.replace("\n", "\n  "))
    print(f"\n✅ FINAL STATUS: {status}")
    return {"user_message": message, "final_status": status,
            "audit_log": audit("CommunicationAgent", "draft_user_message", f"final_status={status}")}

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

    for r in results:
        path = " → ".join(dict.fromkeys(e["agent"].replace("Agent", "").replace("Gate", "")
                                        for e in r["audit_log"]))
        print(f"\n{'═' * 55}\nAUDIT LOG: {r['ticket_number']}  |  {r['final_status']}  |  {path}\n{'═' * 55}")
        for e in r["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19} {e['action']:<19} {e['detail']}")
