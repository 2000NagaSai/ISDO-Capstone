"""
ISDO Lab C2 — Mock Jira Service Management REST API (Flask Shim)
Mimics the Jira REST API for service requests so the MCP server
can make real HTTP calls without touching production.

Endpoints:
  GET  /rest/agile/1.0/board/requests   — list requests (filters: request_type, priority, assignee, status)
  GET  /rest/api/2/issue?request_type=Access+Grant — same list, filtered
  GET  /rest/api/2/issue/<key>          — one request in Jira-style nested 'fields'
  PUT  /rest/api/2/issue/<key>          — update a request (flat or Jira-style body)
  POST /rest/api/2/issue                — create a request
  GET  /health                          — health check

Run from the project root:  python mcp_server/jira_shim.py      (port 5002)
"""

from flask import Flask, jsonify, request
import csv
import os

app = Flask(__name__)
DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "requests.csv"))
FILTERS = ["request_type", "priority", "assignee", "status"]
# Jira field name -> flat CSV column
JIRA_TO_FLAT = {"issuetype": "request_type", "customfield_sla": "sla"}


def load_requests():
    data = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                if None in row or None in row.values():
                    print(f"Warning: skipping malformed row on line {line_no} of {DATA_FILE}")
                    continue
                data[row["key"]] = {k: v.strip() for k, v in row.items()}
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return data


REQUESTS = load_requests()


def flatten(fields):
    """Turn Jira-style {"status": {"name": "Done"}} into flat {"status": "Done"}."""
    flat = {}
    for k, v in fields.items():
        if isinstance(v, dict):
            v = v.get("name", v.get("displayName", ""))
        flat[JIRA_TO_FLAT.get(k, k)] = v
    return flat


def next_key():
    nums = [int(k.split("-")[1]) for k in REQUESTS if k.startswith("REQ-") and k.split("-")[1].isdigit()]
    return f"REQ-{max(nums, default=1000) + 1}"


@app.route("/rest/agile/1.0/board/requests", methods=["GET"])
@app.route("/rest/api/2/issue", methods=["GET"])
def list_requests():
    results = list(REQUESTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if str(r.get(key, "")).lower() == val.strip().lower()]
    return jsonify({"issues": results, "total": len(results)})


@app.route("/rest/api/2/issue/<key>", methods=["GET"])
def get_request(key):
    req = REQUESTS.get(key)
    if not req:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify({
        "key": key,
        "fields": {
            "summary": req.get("summary"),
            "priority": {"name": req.get("priority")},
            "status": {"name": req.get("status")},
            "assignee": {"displayName": req.get("assignee")},
            "customfield_sla": req.get("sla"),
            "issuetype": {"name": req.get("request_type")},
        },
    })


@app.route("/rest/api/2/issue/<key>", methods=["PUT"])
def update_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({"errorMessages": ["No update body provided"]}), 400
    fields = flatten(data.get("fields", data))
    fields.pop("key", None)
    REQUESTS[key].update(fields)
    print(f"[Jira Mock] Updated {key}: {fields}")
    return jsonify({"key": key, "message": "Updated successfully"})


@app.route("/rest/api/2/issue", methods=["POST"])
def create_request():
    data = request.get_json(silent=True) or {}
    fields = flatten(data.get("fields", data))
    if not fields.get("summary"):
        return jsonify({"errorMessages": ["Field 'summary' is required"]}), 400
    key = next_key()
    REQUESTS[key] = {
        "key": key,
        "summary": fields["summary"],
        "request_type": fields.get("request_type", ""),
        "priority": fields.get("priority") or "Medium",
        "assignee": fields.get("assignee", ""),
        "sla": fields.get("sla", ""),
        "status": "Open",
    }
    print(f"[Jira Mock] Created request: {key}")
    return jsonify({"key": key, "message": "Request created"}), 201


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print("Jira Mock API starting on http://localhost:5002")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    print("Endpoints: GET /rest/agile/1.0/board/requests  |  GET /health")
    app.run(port=5002, debug=True)