"""
ISDO Lab C4 — Resolution / KB Agent
Searches the ChromaDB knowledge base (built in Lab C1) and drafts a resolution.
HIGH confidence on a non-P1 ticket -> auto-resolve. Anything else -> HITL flag.

Run from the project root:   python agents/resolution_agent.py
Needs: Lab C1 run once (data/chroma_db with collection 'isdo_kb'),
       ANTHROPIC_API_KEY in the project's .env file.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # agents/ -> project root
KB_DIR = PROJECT_ROOT / "data" / "kb"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "isdo_kb"

load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
# TEMPERATURE = 0.0
MAX_TOKENS = 1500
MAX_LOOP_TURNS = 4

# Confidence thresholds on similarity score (1 - cosine distance). Tune after your first run.
HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.35
# Priorities allowed to auto-resolve. P1 never auto-resolves.
# (Lab text says "P3/P4" in Step 3 but expects the P2 VPN ticket to auto-resolve — P2 is included here.)
AUTO_RESOLVE_PRIORITIES = {"P2", "P3", "P4"}

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(f"ANTHROPIC_API_KEY not set. Add it to {PROJECT_ROOT / '.env'}")

client = anthropic.Anthropic()

# ── CONNECT TO THE LAB C1 KNOWLEDGE BASE ──────────────────────────────────────
def load_kb():
    """Open the persistent ChromaDB collection built in Lab C1 (does not rebuild it)."""
    db = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        kb = db.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit(f"Collection '{COLLECTION_NAME}' not found in {DB_DIR}.\n"
                 f"Run Lab C1 first:  python labs/C1/kb_setup.py")
    if kb.count() == 0:
        sys.exit(f"Collection '{COLLECTION_NAME}' is empty. Re-run: python labs/C1/kb_setup.py")
    print(f"KB connected: {kb.count()} chunks in '{COLLECTION_NAME}' ({DB_DIR})")
    return kb


KB = load_kb()


def score_to_confidence(score):
    if score > HIGH_THRESHOLD:
        return "HIGH"
    if score > MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────
tools = [
    {
        "name": "search_kb",
        "description": "Search the IT knowledge base for articles matching the ticket. "
                       "Returns the top 2 articles with a confidence score (0-1) and the full article text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The ticket's summary and details, used as the search text"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "draft_resolution",
        "description": "Record the resolution for the ticket, based on the KB article found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "resolution_text": {
                    "type": "string",
                    "description": "3-4 numbered steps for the requester, taken from the KB article's "
                                   "Resolution Steps. If no article matches, say the ticket is escalated to L2.",
                },
                "auto_resolve": {
                    "type": "boolean",
                    "description": "True only if confidence is HIGH, priority is not P1, and the article's "
                                   "Auto-Resolve Eligibility section allows L1 auto-resolution for this case",
                },
                "confidence": {
                    "type": "string", "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": f"From the top article's score: HIGH > {HIGH_THRESHOLD}, "
                                   f"MEDIUM > {MEDIUM_THRESHOLD}, otherwise LOW",
                },
                "kb_article_used": {"type": "string",
                                    "description": "File name of the KB article used, or 'none'"},
            },
            "required": ["ticket_number", "resolution_text", "auto_resolve", "confidence", "kb_article_used"],
        },
    },
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────
def search_kb(query, n_articles=2):
    """Query ChromaDB, keep the best chunk per article, return the top articles in full."""
    raw = KB.query(query_texts=[query], n_results=min(10, KB.count()))
    best = {}
    for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
        # Lab C1 script stores 'article'; the lab's sample kb_setup stores 'filename'
        fname = meta.get("article") or meta.get("filename") or "unknown"
        score = max(0.0, min(1.0, 1.0 - dist))
        best[fname] = max(best.get(fname, 0.0), score)

    articles = []
    for fname, score in sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:n_articles]:
        path = KB_DIR / fname
        articles.append({
            "article": fname,
            "confidence_score": round(score, 2),
            "confidence": score_to_confidence(score),
            "content": path.read_text(encoding="utf-8") if path.exists() else "",
        })
    return {"query": query, "articles": articles}


def apply_guardrails(draft, scores, priority):
    """Code-enforced HITL boundary. The model can make the decision stricter, never looser."""
    article = draft.get("kb_article_used", "none")
    top_score = max(scores.values(), default=0.0)
    score = scores.get(article, top_score if article in ("", "none") else 0.0)
    confidence = score_to_confidence(score)

    reasons = []
    if confidence != "HIGH":
        reasons.append(f"{confidence} confidence ({score:.0%})")
    if priority not in AUTO_RESOLVE_PRIORITIES:
        reasons.append(f"priority {priority} requires human handling")
    if not draft.get("auto_resolve"):
        reasons.append("KB article / agent does not allow auto-resolution")

    return {
        **draft,
        "priority": priority,
        "score": round(score, 2),
        "confidence": confidence,
        "model_confidence": draft.get("confidence"),
        "auto_resolve": not reasons,
        "hitl_reasons": reasons,
    }

# ── RESOLUTION AGENT ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are the ISDO Resolution Agent for Zensar's IT Service Desk.

For each ticket:
1. Call search_kb ONCE, using the ticket summary and details as the query. Do not retry
   with other wording — a LOW score is a valid result for topics the KB does not cover.
2. Call draft_resolution ONCE:
   - resolution_text: 3-4 numbered steps copied closely from the matched article's
     "Resolution Steps" (exact tools, paths, URLs). No generic advice.
   - confidence: from the top article's score (HIGH > {HIGH_THRESHOLD}, MEDIUM > {MEDIUM_THRESHOLD}, else LOW).
   - auto_resolve: true only if confidence is HIGH, the priority is not P1, and the
     article's "Auto-Resolve Eligibility" section allows it for this situation
     (e.g. multi-user outages are never auto-resolved).
   - If confidence is LOW, set kb_article_used to "none" and say the ticket is escalated to L2.
3. Then reply with one short line and stop."""


def resolve_ticket(ticket_number, short_description, description, category, priority):
    """Run the resolution agent on one ticket. Returns the guarded resolution dict (or None).
    category/priority normally come from the Triage Agent (Lab C3)."""
    print(f"\n{'=' * 55}\nResolving: {ticket_number} | Category: {category} | Priority: {priority}\n{'=' * 55}")
    print(f"Issue: {short_description}")

    messages = [{
        "role": "user",
        "content": f"Find a resolution for this ticket:\n\nTicket: {ticket_number}\n"
                   f"Category: {category}\nPriority: {priority}\n"
                   f"Summary: {short_description}\nDetails: {description}",
    }]
    scores, result = {}, None

    for _ in range(MAX_LOOP_TURNS):
        response = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, 
            system=SYSTEM_PROMPT, tools=tools, messages=messages,
        )
        if response.stop_reason != "tool_use":
            if response.stop_reason != "end_turn":
                print(f"  [stopped: {response.stop_reason}]")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "search_kb":
                out = search_kb(block.input.get("query", short_description))
                print(f"  -> KB search: '{out['query'][:70]}'")
                for art in out["articles"]:
                    scores[art["article"]] = art["confidence_score"]
                    print(f"     [{art['confidence_score']:.0%}] {art['article']}")
            elif block.name == "draft_resolution":
                out = result = apply_guardrails({**block.input, "ticket_number": ticket_number},
                                                scores, priority)
            else:
                out = {"error": f"Unknown tool: {block.name}"}
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(out)})
        messages.append({"role": "user", "content": tool_results})
        if result:
            break                     # decision recorded — no need for another model turn
    else:
        print(f"  [loop cap of {MAX_LOOP_TURNS} turns reached]")

    if result is None:
        result = {"ticket_number": ticket_number, "priority": priority, "confidence": "LOW",
                  "auto_resolve": False, "kb_article_used": "none", "resolution_text": "",
                  "hitl_reasons": ["agent did not produce a resolution"]}

    print(f"\n  -> Confidence: {result['confidence']}  |  Auto-resolve: {result['auto_resolve']}")
    if result.get("model_confidence") and result["model_confidence"] != result["confidence"]:
        print(f"     (model said {result['model_confidence']}; score-based rule applied)")
    print(f"  -> KB Article: {result['kb_article_used']}")
    print("\n  RESOLUTION DRAFT:")
    for line in result.get("resolution_text", "").splitlines():
        print(f"  {line}")
    if not result["auto_resolve"]:
        print(f"\n  WARNING  HITL FLAG: human review required — {'; '.join(result['hitl_reasons'])}")
    return result

# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. Error: authentication failed.",
         "Network", "P2"),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset.",
         "Access", "P2"),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today.",
         "Application", "P1"),
        # Step 5 — uncomment and run again: expect LOW confidence + HITL flag
        ("TEST-WEBEX", "Cisco Webex not launching on Mac M2",
         "Cisco Webex not launching on Mac M2", "Software", "P3"),
    ]

    results = [resolve_ticket(*t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    for r in results:
        status = "AUTO-RESOLVED" if r["auto_resolve"] else "HITL"
        print(f"  {r['ticket_number']:<12} {r['confidence']:<7} {r.get('score', 0):>4.0%}  "
              f"{status:<14} {r['kb_article_used']}")