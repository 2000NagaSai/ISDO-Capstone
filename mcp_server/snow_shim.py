"""
ISDO Lab C2 — Mock ServiceNow REST API (Flask Shim)
Mimics the ServiceNow Table API so the MCP server can make real HTTP calls
without touching a production system.

Endpoints:
  GET   /api/now/table/incident            — list incidents (filters: category, priority, state, assignment_group)
  GET   /api/now/table/incident/<number>   — get one incident
  PATCH /api/now/table/incident/<number>   — update fields in memory (e.g. state, work_notes)
  POST  /api/now/table/incident            — create an incident
  GET   /health                            — health check

Run from the project root:  python mcp_server/snow_shim.py      (port 5001)
"""

from flask import Flask, jsonify, request
import csv
import os

app = Flask(__name__)
DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "incidents.csv"))
FILTERS = ["category", "state", "priority", "assignment_group"]


def load_incidents():
    incidents = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_no, row in enumerate(reader, start=2):
                # A row with extra/missing columns (e.g. unquoted commas) is skipped, not loaded half-shifted
                if None in row or None in row.values():
                    print(f"Warning: skipping malformed row on line {line_no} of {DATA_FILE}")
                    continue
                incidents[row["number"]] = {k: v.strip() for k, v in row.items()}
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return incidents


INCIDENTS = load_incidents()  # in-memory store for this session


@app.route("/api/now/table/incident", methods=["GET"])
def list_incidents():
    results = list(INCIDENTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if str(r.get(key, "")).lower() == val.strip().lower()]
    return jsonify({"result": results, "total": len(results)})


@app.route("/api/now/table/incident/<number>", methods=["GET"])
def get_incident(number):
    incident = INCIDENTS.get(number)
    if not incident:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.route("/api/now/table/incident/<number>", methods=["PATCH"])
def update_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not isinstance(updates, dict) or not updates:
        return jsonify({"error": "Request body must be a non-empty JSON object"}), 400
    updates.pop("number", None)  # the record key cannot be changed
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.route("/api/now/table/incident", methods=["POST"])
def create_incident():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get("number"):
        return jsonify({"error": "Missing required field: number"}), 400
    if data["number"] in INCIDENTS:
        return jsonify({"error": f"Incident {data['number']} already exists"}), 409
    INCIDENTS[data["number"]] = data
    print(f"[ServiceNow Mock] Created incident: {data['number']}")
    return jsonify({"result": data, "message": "Incident created"}), 201


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print("ServiceNow Mock API starting on http://localhost:5001")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    print("Endpoints: GET /api/now/table/incident  |  GET /health")
    app.run(port=5001, debug=True)