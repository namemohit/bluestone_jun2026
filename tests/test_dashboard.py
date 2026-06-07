"""Dashboard backend: value-based corrections, params + training endpoints, retrain/promote."""
from fastapi.testclient import TestClient

from dashboard.app import app, state


def test_state_shape():
    state.reset()
    client = TestClient(app)
    s = client.get("/api/state").json()
    assert {"report", "score", "cards", "params", "training"} <= set(s)
    assert s["training"]["settings"]["target"] == 0.95
    # pipeline params expose live + config fields
    keys = {f["key"] for step in s["params"] for f in step["fields"]}
    assert {"fusion.sim_threshold", "employee.threshold", "grouping.gap_sec"} <= keys


def test_correct_retrain_promotes():
    state.reset()
    client = TestClient(app)
    s = client.get("/api/state").json()
    assert s["score"] == 0 and len(s["cards"]) == 7

    for card in s["cards"]:  # apply each reviewer's suggested value
        client.post("/api/correct", json={"id": card["id"], "value": card["control"]["suggested"]})

    r = client.post("/api/retrain").json()
    assert r["result"]["before"] == 0 and r["result"]["after"] == 100
    assert r["result"]["promoted"] is True
    final = r["state"]
    assert final["score"] == 100 and len(final["cards"]) == 0
    assert final["report"]["customers"]["unique_customers"] == 7
    assert final["report"]["employees"]["headcount"] == 2


def test_params_live_update_changes_value():
    state.reset()
    client = TestClient(app)
    s = client.post("/api/params", json={"patch": {"employee.threshold": 0.5}}).json()
    emp = next(f for step in s["params"] for f in step["fields"] if f["key"] == "employee.threshold")
    assert emp["value"] == 0.5


def test_train_settings_and_manual_promote():
    state.reset()
    client = TestClient(app)
    # turn auto-promote off -> eligible candidate awaits approval
    client.post("/api/train_settings", json={"auto_promote": False})
    s = client.get("/api/state").json()
    for card in s["cards"]:
        client.post("/api/correct", json={"id": card["id"], "value": card["control"]["suggested"]})
    r = client.post("/api/retrain").json()
    assert r["result"]["eligible"] is True and r["result"]["promoted"] is False
    ver = r["result"]["candidate_version"]
    promoted = client.post("/api/promote", json={"version": ver}).json()
    assert promoted["state"]["active_version"] == ver
    assert promoted["state"]["score"] == 100


def test_employee_naming_records_directory():
    state.reset()
    client = TestClient(app)
    s = client.get("/api/state").json()
    emp = next(c for c in s["cards"] if c["kind"] == "employee")
    sug = emp["control"]["suggested"]
    client.post("/api/correct", json={"id": emp["id"], "value": {**sug, "name": "Ramesh"}})
    s2 = client.get("/api/state").json()
    assert "Ramesh" in s2["employees"].values()


def test_sources_preview_builds_and_masks_url():
    state.sources = {"nvr": {"brand": "hikvision", "host": "10.0.0.5", "rtsp_port": 554,
                             "username": "admin", "password": "secret"},
                     "cameras": [{"id": "cam_entry", "role": "entry", "channel": 1, "stream": "main"}]}
    client = TestClient(app)
    r = client.get("/api/sources").json()
    assert r["preview"][0]["url"].endswith("/Streaming/Channels/101")
    assert "secret" not in r["preview"][0]["url"]  # credentials masked
