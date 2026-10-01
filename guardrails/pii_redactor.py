"""
ISDO Lab C9 — PII Redaction Middleware
Masks PII before any ticket data is sent to Claude.

Detects: person names (spaCy NER + context rules), usernames / login IDs, email addresses,
employee IDs, IP addresses and phone numbers. Ticket references (INC/REQ/CHG) are kept.

Names are detected in two layers so they are caught even if the spaCy model is missing:
  1. spaCy NER (PERSON)                    — needs:  python -m spacy download en_core_web_sm
  2. Context rules (always on) — a capitalised name after words like User, Employee,
     Contractor, Mr/Ms/Dr, Dear, Contact, "reported by", "for" ...
Usernames are detected from DOMAIN\\user, a label ("username: jsmith", "login id: x"), or a
username-shaped token after user/login ("user jsmith01", "login j.smith").
Known limit: a bare lowercase username with no label, dot, underscore or digit
("user jsmith reports ...") is not detected — it can't be told apart from an ordinary word.

Usage:
    from guardrails.pii_redactor import redact, restore

    clean_text, mapping = redact(raw_text)
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # guardrails/ -> project root

# ── spaCy (optional) ──────────────────────────────────────────────────────────
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠  spaCy is installed but the model 'en_core_web_sm' is missing — using rule-based name "
              "detection only.\n   Install it with:  python -m spacy download en_core_web_sm")
except ImportError:
    print("⚠  spaCy not installed — using rule-based name detection only.")
SPACY_AVAILABLE = nlp is not None

# ── PATTERNS ──────────────────────────────────────────────────────────────────
_NAME_WORD = r"[A-Z][a-z]+(?:['\-][A-Z]?[a-z]+)?"                 # John, D'Souza, Mary-Ann, O'Neil
_NAME_WORD_CAP = r"(?:[A-Z]'[A-Z][a-z]+|" + _NAME_WORD + r")"     # also D'Souza style
_FULL_NAME = rf"{_NAME_WORD_CAP}(?:\s+{_NAME_WORD_CAP}){{1,2}}"    # 2-3 words
_ANY_NAME = rf"{_NAME_WORD_CAP}(?:\s+{_NAME_WORD_CAP}){{0,2}}"     # 1-3 words

# Words that strongly announce a person -> a single capitalised word is enough ("Dear Ravi")
_STRONG_CUES = (r"Mr|Mrs|Ms|Miss|Dr|Dear|Hi|Hello|Regards|Thanks|Contact|Name|Employee|Contractor|"
                r"Colleague|Manager|Requester|Requestor|Reported\s+by|Requested\s+by|Raised\s+by|"
                r"Submitted\s+by|Assigned\s+to|On\s+behalf\s+of|Approved\s+by|cc")
# Weaker cues -> need a full (2-3 word) name ("User John Smith", "reset for Michael D'Souza")
_WEAK_CUES = r"User|for|from|with|to|by|and|ask|call|email|tell|inform"

NAME_RULES = [   # (pattern, minimum words the cleaned name must still have)
    (re.compile(rf"(?i:\b(?:{_STRONG_CUES}))\.?[:,]?\s+(?P<pii>{_ANY_NAME})"), 1),
    (re.compile(rf"(?i:\b(?:{_WEAK_CUES}))[:,]?\s+(?P<pii>{_FULL_NAME})"), 2),
]
LEADING_FILLERS = {"the", "a", "an", "my", "our", "your", "their", "his", "her"}

# Capitalised words that are never (part of) a person's name in ticket text
NOT_NAMES = {
    "the", "a", "an", "this", "that", "my", "our", "your", "all", "multiple", "new", "senior",
    "user", "users", "team", "teams", "department", "dept", "finance", "marketing", "sales", "hr",
    "it", "support", "desk", "service", "network", "application", "server", "access", "email",
    "hardware", "software", "building", "floor", "room", "board", "project", "phoenix", "office",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "windows", "outlook", "exchange", "teams", "zoom", "webex",
    "cisco", "anyconnect", "adobe", "acrobat", "sharepoint", "oracle", "python", "microsoft",
    "azure", "active", "directory", "password", "reset", "account", "laptop", "printer", "error",
    "ticket", "request", "issue", "incident", "admin", "administrator", "manager", "engineer",
    "contractor", "employee", "vpn", "sap", "erp", "crm", "salesforce", "mac", "macbook", "sonoma",
}

USERNAME_RULES = [
    # DOMAIN\username   e.g. ZENSAR\jsmith
    re.compile(r"\b(?P<pii>[A-Za-z][A-Za-z0-9\-]{1,14}\\[A-Za-z][\w.\-]{1,30})"),
    # keyword + separator + anything:  "username: jsmith", "login=p.chary", "user id - jsmith"
    re.compile(r"(?i:\b(?:user\s*name|user\s*id|userid|login(?:\s*id)?|sam\s*account\s*name|"
               r"account\s*name|ad\s*account|network\s*id|windows\s*id|logon\s*name))"
               r"\s*[:=\-]\s*(?P<pii>[A-Za-z][\w.\-]{1,30}[A-Za-z0-9])"),
    # keyword + username-looking token (has a dot, underscore or digit): "user jsmith01", "login j.smith"
    re.compile(r"(?i:\b(?:user|username|user\s*id|userid|login|account))\s+"
               r"(?P<pii>[a-z][a-z0-9]*[._\d][\w.\-]*[a-z0-9])\b"),
]

REGEX_PATTERNS = [   # (label, compiled pattern) — the whole match is the PII value
    ("EMAIL",       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("IP_ADDRESS",  re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("EMPLOYEE_ID", re.compile(r"\b(?:EMP|ZEN)[\-\s]?\d{3,6}\b", re.IGNORECASE)),
    ("PHONE",       re.compile(r"(?<![\w+])(?:\+91[\-\s]?)?\d{10}(?!\d)"         # +91-9876543210 / 9876543210
                               r"|(?<![\w+])(?:\+\d{1,3}[\-\s])?\(?\d{3}\)?[\-\s]\d{3}[\-\s]\d{4}(?!\d)")),
]
TICKET_REF = re.compile(r"\b(?:INC|REQ|CHG|TASK|RITM)-?\d{4,7}\b", re.IGNORECASE)   # never PII

# ── AUDIT LOGGER (module-level, in memory) ────────────────────────────────────
audit_log = []


def _audit(action, detail):
    entry = {"timestamp": datetime.now().isoformat(), "module": "PIIRedactor",
             "action": action, "detail": detail}
    audit_log.append(entry)
    return entry

# ── DETECTION ─────────────────────────────────────────────────────────────────
def _clean_name(value, min_words=1):
    """Return the name if it looks like a person's name, else None.
    'The John Smith' -> 'John Smith';  'Priya Sharma Team' -> 'Priya Sharma';
    'Adobe Acrobat Pro' / 'Finance Team' -> None (contains words that are never names)."""
    words = value.split()
    while words and words[0].lower() in LEADING_FILLERS:
        words.pop(0)
    while words and words[-1].lower() in NOT_NAMES:
        words.pop()
    if len(words) < min_words or any(w.lower() in NOT_NAMES for w in words):
        return None
    return " ".join(words)


def _find_spans(text):
    """Return a list of (start, end, label) PII spans in the ORIGINAL text."""
    spans = []

    for label, pattern in REGEX_PATTERNS:
        spans += [(m.start(), m.end(), label) for m in pattern.finditer(text)]

    for rule in USERNAME_RULES:
        for m in rule.finditer(text):
            if "@" not in m.group("pii"):                       # emails handled above
                spans.append((m.start("pii"), m.end("pii"), "USERNAME"))

    names = set()
    if SPACY_AVAILABLE:
        for ent in nlp(text).ents:
            if ent.label_ == "PERSON":
                name = _clean_name(ent.text.strip(" .,;:()"))
                # short all-caps tokens (VPN, SLA, PII) are acronyms spaCy sometimes tags PERSON
                if name and not (name.isupper() and len(name) <= 5):
                    names.add(name)
    for rule, min_words in NAME_RULES:
        for m in rule.finditer(text):
            name = _clean_name(m.group("pii"), min_words)
            if name:
                names.add(name)
    # redact EVERY occurrence of each detected name, not just the one next to the cue word
    for name in names:
        spans += [(m.start(), m.end(), "NAME") for m in re.finditer(rf"\b{re.escape(name)}\b", text)]

    # ticket references are never PII: drop anything overlapping them
    protected = [(m.start(), m.end()) for m in TICKET_REF.finditer(text)]
    spans = [s for s in spans if not any(s[0] < pe and ps < s[1] for ps, pe in protected)]

    # resolve overlaps: earliest start wins, then the longest span
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    result, last_end = [], -1
    for s in spans:
        if s[0] >= last_end:
            result.append(s)
            last_end = s[1]
    return result


def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns:
      - clean_text: text with PII replaced by tokens like [EMAIL_1], [NAME_1]
      - mapping: dict to restore original values later

    The same value always gets the same token within one call.

    Example:
      clean, m = redact("Contact john.doe@corp.com or call 9876543210")
      # clean  = "Contact [EMAIL_1] or call [PHONE_1]"
      # m      = {"[EMAIL_1]": "john.doe@corp.com", "[PHONE_1]": "9876543210"}
    """
    mapping, value_to_token, counters = {}, {}, {}
    parts, pos = [], 0
    for start, end, label in _find_spans(text):
        value = text[start:end]
        key = (label, value.lower())
        if key not in value_to_token:
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            value_to_token[key] = token
            mapping[token] = value
        parts += [text[pos:start], value_to_token[key]]
        pos = end
    clean = "".join(parts) + text[pos:]

    if mapping:
        _audit("redact", f"{len(mapping)} PII item(s) masked: {list(mapping.keys())}")   # tokens only, never values
    else:
        _audit("redact", "No PII detected")
    return clean, mapping


def contains_pii(text: str) -> bool:
    return bool(_find_spans(text))


def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored


def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# ── AUDIT TRAIL LOGGER ────────────────────────────────────────────────────────
class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval.
    The rationale is PII-redacted before it is printed or written to disk."""

    def __init__(self, log_file: str = "logs/audit_trail.jsonl"):
        path = Path(log_file)
        self.log_file = path if path.is_absolute() else PROJECT_ROOT / path   # same place from any folder
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        # Mask PII with the label only (e.g. [EMAIL]) — no mapping is kept for the audit file
        out, pos = [], 0
        for s, e, label in _find_spans(rationale or ""):
            out += [rationale[pos:s], f"[{label}]"]
            pos = e
        safe_rationale = ("".join(out) + (rationale or "")[pos:])[:200]
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": safe_rationale,
            "approval_status": approval_status,
        }
        self.entries.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'=' * 55}\nFULL AUDIT TRAIL ({len(self.entries)} entries)\n{'=' * 55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(f"PII REDACTION DEMO   (spaCy NER: {'ON' if SPACY_AVAILABLE else 'OFF - rule-based names only'})")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
        "AD account locked for username: jsmith01. Login ZENSAR\\p.chary also failing. Reported by Naga Sai.",
        "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today.",
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 — VPN failure after password change", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK — 90 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to user John Smith at john.smith@zensar.com", "Auto")   # PII is masked in the file

    logger.print_trail()
    print(f"\nAudit log saved to: {logger.log_file}")
