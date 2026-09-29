"""
ISDO Lab C6 — LangGraph Orchestrator: wires the ISDO agents into one StateGraph.

    triage -> resolution -> sla --(hitl_required)--> hitl -> communication -> END
                                 \\------(otherwise)-------> communication -> END

Reuses the agents from Labs C3-C5 (agents/triage_agent.py, resolution_agent.py, sla_agent.py).

Run from the project root:   python orchestrator/supervisor.py
Needs: ANTHROPIC_API_KEY in .env, Lab C1 KB (data/chroma_db), snow_shim running (Lab C2).
"""

import operator
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

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

# ── SHARED STATE ──────────────────────────────────────────────────────────────
class TicketState(TypedDict, total=False):
    # input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
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
    return {
        "kb_article": r.get("kb_article_used", "none"),
        "resolution_text": r.get("resolution_text", ""),
        "auto_resolve": bool(r.get("auto_resolve")),
        "confidence": r.get("confidence", "LOW"),
        "audit_log": audit("ResolutionAgent", "search_kb",
                           f"{r.get('kb_article_used')} | {r.get('confidence')} ({r.get('score', 0):.0%}) | "
                           f"auto_resolve={r.get('auto_resolve')}"),
    }


def sla_node(state: TicketState) -> dict:
    print("\n▶ SLA AGENT — checking deadline")
    priority = effective_priority(state)
    s = sla_agent.get_sla_status(state["ticket_number"], state["sla_due"], priority)
    if "error" in s:
        print(f"  SLA check failed: {s['error']}")
        s = {"breach_risk": "UNKNOWN", "minutes_remaining": None, "requires_escalation": False}
    print(f"  SLA Risk: {s['breach_risk']}  |  Minutes remaining: {s['minutes_remaining']}")

    escalation_required = bool(s["requires_escalation"])
    hitl_required = escalation_required and priority in sla_agent.HITL_PRIORITIES
    entries = audit("SLAAgent", "get_sla_status",
                    f"{priority} {s['breach_risk']}, {s['minutes_remaining']} min left; "
                    f"escalation_required={escalation_required}, hitl_required={hitl_required}")

    # Non-P1 escalations (e.g. breached P2) go ahead without a human; P1 waits for the HITL node
    if escalation_required and not hitl_required:
        team = sla_agent.ESCALATION_TEAMS.get(state.get("triage_category"), sla_agent.DEFAULT_TEAM)
        sla_agent.update_ticket(state["ticket_number"], "escalate", team,
                                note=f"Auto-escalated: SLA {s['breach_risk']}")
        entries += audit("SLAAgent", "update_ticket", f"auto-escalated to {team}")

    return {"sla_breach_risk": s["breach_risk"], "escalation_required": escalation_required,
            "hitl_required": hitl_required, "auto_resolve": state.get("auto_resolve", False) and not hitl_required,
            "audit_log": entries}


def hitl_node(state: TicketState) -> dict:
    print("\n▶ HITL GATE — human approval required")
    team = sla_agent.ESCALATION_TEAMS.get(state.get("triage_category"), sla_agent.DEFAULT_TEAM)
    approved = sla_agent.hitl_approve(state["ticket_number"], "Escalate ticket",
                                      f"Escalate to {team} (SLA {state.get('sla_breach_risk')})")
    entries = audit("HITLGate", "human_decision", f"{'APPROVED' if approved else 'REJECTED'}: escalate to {team}")
    if approved:
        sla_agent.update_ticket(state["ticket_number"], "escalate", team,
                                note=f"Escalation approved by human operator (SLA {state.get('sla_breach_risk')})")
        entries += audit("SLAAgent", "update_ticket", f"escalated to {team}")
    else:
        sla_agent.update_ticket(state["ticket_number"], "add_note",
                                note="Escalation declined by human operator")
    return {"hitl_approved": approved, "audit_log": entries}


def communication_node(state: TicketState) -> dict:
    print("\n▶ COMMUNICATION AGENT")
    if state.get("auto_resolve"):
        kind, status = "self-service resolution with the steps below", "RESOLVED"
    elif state.get("hitl_approved"):
        kind, status = "escalation confirmation (a senior team has been engaged)", "ESCALATED"
    elif state.get("escalation_required") and not state.get("hitl_required"):
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
            system="You write short, polite IT service desk messages to end users. Plain text, no markdown "
                   "headings, under 120 words. Start with 'Dear User,'. Never include names, emails or IDs of "
                   "people. Do not promise times that are not given.",
            messages=[{"role": "user", "content": f"Write a {kind} for this ticket.\n\n{facts}"}],
        )
        message = "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as e:           # never lose the ticket because the message draft failed
        print(f"  (message draft failed: {e}; using template)")
        message = f"Dear User, regarding {state['ticket_number']} ({state['short_description']}): {kind}."

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
    return graph.invoke({**ticket, "audit_log": []})

# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated now = 2024-01-15 10:30 (Lab C5). Same demo SLA times as Lab C5.
    test_tickets = [
        {   # P2, 90 min left (AT_RISK) -> Triage -> Resolution -> SLA -> Communication
            "ticket_number": "INC0001001",
            "short_description": "VPN not connecting after password change",
            "description": "User reports VPN client fails to connect after AD password was reset. "
                           "Error: authentication failed.",
            "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00",
        },
        {   # P1, 10 min left (CRITICAL) -> ... -> SLA -> HITL (type y) -> Communication
            "ticket_number": "INC0001002",
            "short_description": "Cannot access ERP system - login error",
            "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
                           "Started 09:00 today.",
            "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00",
        },
    ]

    results = [process_ticket(t) for t in test_tickets]

    for r in results:
        path = " → ".join(dict.fromkeys(e["agent"].replace("Agent", "").replace("Gate", "")
                                        for e in r["audit_log"]))
        print(f"\n{'═' * 55}\nAUDIT LOG: {r['ticket_number']}  |  {r['final_status']}  |  {path}\n{'═' * 55}")
        for e in r["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19} {e['action']:<19} {e['detail']}")