"""
ISDO Lab C5 — SLA & Escalation Agent
Checks SLA breach risk, escalates CRITICAL/BREACHED P1/P2 tickets, and pauses at a
human-in-the-loop (HITL) gate before ANY P1 escalation.

Run from the project root:   python agents/sla_agent.py
Needs: ANTHROPIC_API_KEY in .env. ServiceNow shim (python mcp_server/snow_shim.py) should be
running so update_ticket really PATCHes the mock API; if it isn't, updates are simulated.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = PROJECT_ROOT / "logs" / "hitl_audit.jsonl"
SNOW_URL = "http://localhost:5001/api/now/table/incident"

load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
# TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_LOOP_TURNS = 6

SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)          # fixed "now" for reproducible demos
SLA_TARGET_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
CRITICAL_PCT = 0.20        # < 20% of SLA time left -> CRITICAL
AT_RISK_PCT = 0.50         # < 50% of SLA time left -> AT_RISK
ESCALATE_RISKS = {"BREACHED", "CRITICAL"}
ESCALATE_PRIORITIES = {"P1", "P2"}
HITL_PRIORITIES = {"P1"}   # every escalation of these needs a human "y"

ESCALATION_TEAMS = {
    "Network": "L2-Network-Ops",
    "Application": "L2-App-Support",
    "Server": "L2-Server-Ops",
    "Access": "L2-Security-Ops",
}
DEFAULT_TEAM = "L2-Service-Desk"

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(f"ANTHROPIC_API_KEY not set. Add it to {PROJECT_ROOT / '.env'}")

client = anthropic.Anthropic()

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────
tools = [
    {
        "name": "get_sla_status",
        "description": "Check a ticket's SLA: minutes remaining, breach_risk "
                       "(BREACHED / CRITICAL / AT_RISK / ON_TRACK) and whether escalation is required.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {"type": "string", "description": "SLA due datetime, format YYYY-MM-DD HH:MM:SS"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["ticket_number", "sla_due", "priority"],
        },
    },
    {
        "name": "update_ticket",
        "description": "Update the ticket in ServiceNow: escalate it to a team, add a work note, or change its state. "
                       "P1 escalations are held for human approval and may be rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"]},
                "escalation_team": {"type": "string", "description": "Team to escalate to (for action=escalate)"},
                "note": {"type": "string", "description": "Work note text (for add_note, optional for escalate)"},
                "new_state": {"type": "string",
                              "description": "New state, e.g. In Progress, Escalated, Resolved (for update_state)"},
            },
            "required": ["ticket_number", "action"],
        },
    },
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────
def get_sla_status(ticket_number, sla_due, priority, now=SIMULATED_NOW):
    """Pure calculation of SLA breach risk."""
    try:
        due = datetime.strptime(sla_due.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return {"error": f"Invalid sla_due format: {sla_due!r} (expected YYYY-MM-DD HH:MM:SS)"}
    if priority not in SLA_TARGET_MINUTES:
        return {"error": f"Unknown priority: {priority}"}

    target = SLA_TARGET_MINUTES[priority]
    minutes = int((due - now).total_seconds() // 60)
    pct_left = minutes / target

    if minutes < 0:
        risk, msg = "BREACHED", f"SLA breached by {abs(minutes)} minutes"
    elif pct_left < CRITICAL_PCT:
        risk, msg = "CRITICAL", f"Only {minutes} minutes remaining -- breach imminent"
    elif pct_left < AT_RISK_PCT:
        risk, msg = "AT_RISK", f"{minutes} minutes remaining -- at risk"
    else:
        risk, msg = "ON_TRACK", f"{minutes} minutes remaining -- on track"

    return {
        "ticket_number": ticket_number, "priority": priority, "sla_due": sla_due,
        "sla_target_minutes": target, "minutes_remaining": minutes,
        "pct_time_remaining": round(max(pct_left, 0) * 100),
        "breach_risk": risk, "status_message": msg,
        "requires_escalation": risk in ESCALATE_RISKS and priority in ESCALATE_PRIORITIES,
    }


def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """PATCH the ServiceNow mock (Lab C2). Falls back to a simulated update if the shim is down."""
    if action == "escalate":
        patch = {"state": "Escalated", "assignment_group": escalation_team}
        if note:
            patch["work_notes"] = note
        label = f"ESCALATED {ticket_number} -> {escalation_team}"
    elif action == "add_note":
        patch, label = {"work_notes": note or ""}, f"NOTE ADDED to {ticket_number}"
    elif action == "update_state":
        patch, label = {"state": new_state}, f"STATE CHANGED {ticket_number} -> {new_state}"
    else:
        return {"success": False, "error": f"Unknown action: {action}"}

    result = {"ticket_number": ticket_number, "action": action, "fields": patch,
              "timestamp": datetime.now().isoformat(timespec="seconds")}
    try:
        r = requests.patch(f"{SNOW_URL}/{ticket_number}", json=patch, timeout=5)
        result.update(success=r.ok, source="snow_shim", http_status=r.status_code)
        if not r.ok:
            result["error"] = r.json().get("error", r.text)
    except requests.exceptions.ConnectionError:
        result.update(success=True, source="simulated (snow_shim not running)")

    print(f"  [ServiceNow Mock] {label}" + ("" if result["success"] else f"  FAILED: {result.get('error')}"))
    return result

# ── HITL GATE ─────────────────────────────────────────────────────────────────
def audit(event):
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    event["logged_at"] = datetime.now().isoformat(timespec="seconds")
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def hitl_approve(ticket_number, action, detail):
    """Pause for a human decision. Anything other than 'y' (incl. no terminal) = rejected."""
    print("\n  !!!  !!!  !!!")
    print("  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}")
    print(f"  Action:  {action}")
    print(f"  Detail:  {detail}")
    print("  !!!  !!!  !!!")
    try:
        approved = input("  Approve escalation? [y/n]: ").strip().lower() == "y"
    except EOFError:
        approved = False
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")
    audit({"ticket": ticket_number, "action": action, "detail": detail,
           "decision": "APPROVED" if approved else "REJECTED"})
    return approved

# ── SLA AGENT ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status once.
2. If requires_escalation is true (CRITICAL or BREACHED on a P1/P2 ticket), call update_ticket
   with action="escalate" and the team for the ticket's category:
   Network -> L2-Network-Ops, Application -> L2-App-Support, Server -> L2-Server-Ops,
   Access/Security -> L2-Security-Ops, anything else -> {DEFAULT_TEAM}.
   Include a one-line note with the risk level and minutes remaining.
3. If requires_escalation is false, do NOT escalate. For AT_RISK tickets you may add_note a warning.
4. If an escalation is rejected by the human approver, do not retry it: add_note that the
   escalation was declined, then stop.
5. Finish with one short summary line."""


def guard_update(ticket, inp, sla):
    """Code-enforced rules around update_ticket. Returns (tool_input, blocked_result_or_None)."""
    inp = {**inp, "ticket_number": ticket["number"]}          # agent may only touch the current ticket
    if inp.get("action") != "escalate":
        return inp, None
    if not sla or not sla.get("requires_escalation"):
        return inp, {"success": False,
                     "message": "Blocked: escalation only allowed for CRITICAL/BREACHED P1/P2 tickets"}
    inp["escalation_team"] = ESCALATION_TEAMS.get(ticket["category"], DEFAULT_TEAM)
    if ticket["priority"] in HITL_PRIORITIES:
        if not hitl_approve(ticket["number"], "Escalate ticket", f"Escalate to {inp['escalation_team']}"):
            print("  Escalation cancelled by human approver.")
            return inp, {"success": False, "message": "Escalation rejected by human approver"}
    return inp, None


def monitor_ticket(number, short_description, category, priority, sla_due):
    """Run SLA monitoring for one ticket. Returns a summary dict (used by Lab C6)."""
    ticket = {"number": number, "category": category, "priority": priority}
    print(f"\n{'=' * 55}\nSLA Check: {number} | {priority} | Category: {category}\n{'=' * 55}")

    messages = [{"role": "user", "content":
                 f"Monitor SLA for this ticket and escalate if needed:\n\nTicket: {number}\n"
                 f"Description: {short_description}\nCategory: {category}\n"
                 f"Priority: {priority}\nSLA Due: {sla_due}"}]
    sla, escalated, rejected = None, False, False

    for _ in range(MAX_LOOP_TURNS):
        response = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT, tools=tools, messages=messages,
        )
        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            if response.stop_reason != "end_turn":
                print(f"  [stopped: {response.stop_reason}]")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "get_sla_status":
                # Use the ticket's real data, not values the model might have mistyped
                result = sla = get_sla_status(number, sla_due, priority)
                print(f"  -> Risk Level: {sla.get('breach_risk')}")
                print(f"  -> Status:     {sla.get('status_message')}")
            elif block.name == "update_ticket":
                inp, blocked = guard_update(ticket, block.input, sla)
                if blocked:
                    result = blocked
                    rejected |= "rejected" in blocked["message"]
                    if not rejected:
                        print(f"  -> {blocked['message']}")
                else:
                    result = update_ticket(inp["ticket_number"], inp["action"], inp.get("escalation_team"),
                                           inp.get("note"), inp.get("new_state"))
                    escalated |= inp["action"] == "escalate" and result["success"]
            else:
                result = {"error": f"Unknown tool: {block.name}"}
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  [loop cap of {MAX_LOOP_TURNS} turns reached]")

    if sla and sla.get("requires_escalation") and not escalated and not rejected:
        print("  [WARNING] Escalation was required but the agent did not escalate")
    return {"ticket_number": number, "priority": priority,
            "breach_risk": sla.get("breach_risk") if sla else None,
            "minutes_remaining": sla.get("minutes_remaining") if sla else None,
            "escalated": escalated, "escalation_rejected": rejected}

# ── RUN SLA MONITORING ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated now = 2024-01-15 10:30. sla_due values chosen so each ticket lands in the
    # state the lab expects under the Step 1 rules (CRITICAL < 20% left, AT_RISK < 50% left).
    test_tickets = [
        # P1, 10 of 60 min left (17%) -> CRITICAL -> HITL prompt: type y
        ("INC0001002", "Cannot access ERP - SAP login failure", "Application", "P1", "2024-01-15 10:40:00"),
        # P1, 60 min past due -> BREACHED -> HITL prompt: type n
        ("INC0001010", "Exchange server high CPU", "Server", "P1", "2024-01-15 09:30:00"),
        # P2, 90 of 240 min left (37%) -> AT_RISK -> monitored, not escalated
        # Step 5: change 12:00:00 to 10:00:00 -> BREACHED -> auto-escalated to L2-Network-Ops (no HITL for P2)
        ("INC0001001", "VPN not connecting", "Network", "P2", "2024-01-15 12:00:00"),
        # P3, ~46 h left -> ON_TRACK -> monitored only
        ("INC0001003", "Laptop running slowly", "Hardware", "P3", "2024-01-17 09:00:00"),
    ]

    summary = [monitor_ticket(*t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSUMMARY (simulated now: {SIMULATED_NOW:%Y-%m-%d %H:%M})\n{'=' * 55}")
    for s in summary:
        outcome = ("ESCALATED" if s["escalated"] else
                   "ESCALATION REJECTED" if s["escalation_rejected"] else "monitored")
        print(f"  {s['ticket_number']:<12} {s['priority']}  {str(s['breach_risk']):<9} "
              f"{str(s['minutes_remaining']):>6} min  {outcome}")
    print(f"\nHITL decisions logged to {AUDIT_LOG}")