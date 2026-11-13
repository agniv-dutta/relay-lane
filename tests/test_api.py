"""API tests."""

import os

os.environ.setdefault("STANDIN_BASE_URL", "http://localhost:8001")

from fastapi.testclient import TestClient

from relay.api import app


def test_health_shape():
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert "standin_reachable" in d
    assert "results_csv_present" in d
    assert "real_requests_used_today" in d
    assert "real_cap" in d


def test_limits_null_state_not_zero():
    # Force not-measured by pointing probe path absence: limits should not report 0 for unmeasured
    c = TestClient(app)
    r = c.get("/api/limits")
    assert r.status_code == 200
    d = r.json()
    # in_flight and queue_length are untracked -> null + error, never 0
    assert d["in_flight"] is None
    assert d["queue_length"] is None
    assert d["in_flight_error"]
    assert d["queue_length_error"]
    # real_request_cap is our safety cap, always present
    assert d["real_call_cap"]["scope"] == "safety_cap"
    assert d["real_call_cap"]["limit"] is not None
