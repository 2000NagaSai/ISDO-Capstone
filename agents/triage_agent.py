"""
ISDO Lab C3 — Triage Agent
Reads a ticket and assigns: category, priority, assignment group, and PII flag.
Uses the Anthropic SDK with tool calling and a ReAct-style agentic loop.

Run from the project root:   python agents/triage_agent.py
Needs ANTHROPIC_API_KEY in the project's .env file.
"""

import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # agents/ -> project root
INCIDENTS_CSV = PROJECT_ROOT / "data" / "incidents.csv"

load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
# TEMPERATURE = 0.0          # deterministic, rule-based triage
MAX_TOKENS = 1024
MAX_LOOP_TURNS = 5         # safety cap so the loop can never run forever

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(f"ANTHROPIC_API_KEY not set. Add it to {PROJECT_ROOT / '.env'}")

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────
CATEGORIES = ["Network", "Application", "Hardware", "Access", "Email", "Server", "Software"]

tools = [
    {
        "name": "classify_ticket",
        "description": (
            "Record the triage decision for an IT support ticket: category, priority, "
            "assignment group, whether PII is present, and a one-sentence reason. "
            "Call this exactly once per ticket."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES,
                             "description": "The ticket category"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"],
                             "description": "P1=Critical/many users, P2=High/one department, "
                                            "P3=Medium/single user, P4=Low/request"},
                "assignment_group": {"type": "string",
                                     "description": "Team that should own the ticket, e.g. Network-Ops, "
                                                    "App-Support, Desktop-Support, Service-Desk, "
                                                    "Email-Support, Security-Ops, Server-Ops, DBA-Team"},
                "pii_detected": {"type": "boolean",
                                 "description": "True if the ticket text contains a person's name, email "
                                                "address, employee ID, phone number or IP address"},
                "reasoning": {"type": "string",
                              "description": "One sentence explaining the classification decision"},
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"],
        },
    },
    {
        "name": "get_open_tickets",
        "description": "Count the currently Open incidents by category (from the ServiceNow incidents data). "
                       "Use it when current workload or a possible wider outage is relevant.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────
def get_open_tickets(csv_path=INCIDENTS_CSV):
    """Read incidents.csv and return {category: open_count}."""
    counts = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if None in row:                       # malformed row (e.g. unquoted comma) — skip
                    continue
                if row.get("state", "").strip().lower() == "open":
                    cat = row.get("category", "Unknown").strip()
                    counts[cat] = counts.get(cat, 0) + 1
    except FileNotFoundError:
        return {"error": f"File not found: {csv_path}"}
    return counts


def handle_tool_call(tool_name, tool_input):
    """Route a tool call to its implementation."""
    if tool_name == "get_open_tickets":
        return get_open_tickets()          # path is fixed on our side, not chosen by the model
    if tool_name == "classify_ticket":
        return tool_input                  # the structured classification IS the output
    return {"error": f"Unknown tool: {tool_name}"}

# ── TRIAGE AGENT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.

For every ticket you receive, call the classify_ticket tool exactly once to record
category, priority, assignment group, PII flag and a one-sentence reason.
You may call get_open_tickets first if knowing the current open workload helps.
After classify_ticket, reply with one short confirmation line and stop.

Priority rules:
- P1: Service down, many users affected (a whole team, building or site), or security breach
- P2: Significant impact on a single department or function, or a key user blocked with no workaround
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment) or cosmetic issue

PII: set pii_detected=true if the text contains a person's name, email address,
employee ID, phone number or IP address. Placeholders like [REDACTED] are not PII."""


def print_classification(c):
    print(f"  -> Category:    {c.get('category')}")
    print(f"  -> Priority:    {c.get('priority')}")
    print(f"  -> Assign To:   {c.get('assignment_group')}")
    print(f"  -> PII Found:   {c.get('pii_detected')}")
    print(f"  -> Reason:      {c.get('reasoning')}")


def triage_ticket(ticket_number, short_description, description):
    """Run the triage agent on one ticket. Returns the classification dict (or None)."""
    print(f"\n{'=' * 55}\nTriaging: {ticket_number}\n{'=' * 55}")
    print(f"Description: {short_description}")

    messages = [{
        "role": "user",
        "content": f"Please triage this ticket:\n\nTicket: {ticket_number}\n"
                   f"Summary: {short_description}\nDetails: {description}",
    }]
    classification = None

    # Agentic loop: Reason -> Act (tool) -> Observe (tool_result) -> Reason ...
    for _ in range(MAX_LOOP_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            # end_turn (normal finish) or anything unexpected (max_tokens, refusal...) -> stop looping
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
            print(f"  -> Tool called: {block.name}")
            result = handle_tool_call(block.name, block.input)
            if block.name == "classify_ticket":
                classification = {"ticket": ticket_number, **result}
                print_classification(result)
            elif block.name == "get_open_tickets":
                print(f"  -> Open tickets: {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  [loop cap of {MAX_LOOP_TURNS} turns reached]")

    if classification is None:
        print("  [WARNING] Agent finished without calling classify_ticket")
    return classification

# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. Error: authentication failed."),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today."),
        ("INC0001008", "Network switch down - Building C",
         "Network switch in Building C server room unresponsive. 40 users in Building C affected."),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset."),
        ("REQ-1002", "VPN access for new contractor joining project Phoenix",
         "New contractor Ravi Kumar, emp-id ZEN-9823, joining next Monday. Email: ravi.kumar@client.com"),
        # Step 5 — uncomment, run again, then try "Multiple users in Sales cannot access..."
        # ("TEST-006", "Salesforce CRM access issue",
        #  "User cannot access Salesforce CRM from company laptop since this morning."),
    ]

    results = [triage_ticket(*t) for t in test_tickets]

    print(f"\n{'=' * 55}\nOPEN TICKET COUNTS BY CATEGORY\n{'=' * 55}")
    for cat, count in sorted(get_open_tickets().items()):
        print(f"  {cat:<20} {count} open")

    done = sum(r is not None for r in results)
    print(f"\nClassified {done}/{len(test_tickets)} tickets with model {MODEL}")