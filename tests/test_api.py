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
    # model endpoint probe (Prompt E)
    assert "model_endpoint_ok" in d
    assert "model_id_available" in d
    assert "model_base_url" in d
    assert d["model_base_url"].startswith("http")
    assert "key" not in str(d).lower() or "gsk_" not in str(d)


def test_unknown_api_route_is_json_404_not_405():
    c = TestClient(app)
    for method in (c.get, c.post):
        r = method("/api/no-such-route")
        assert r.status_code == 404, r.status_code
        assert r.headers["content-type"].startswith("application/json")
        assert "Unknown API route" in r.json()["detail"]


def test_parse_setting_k_not_matched_inside_chunk():
    """Regression: r"k(\\d+)" matched the k in 'chunk400', setting k=400 (whole corpus)."""
    from relay.api import parse_setting
    for s in ("chunk400_k5_norerank", "chunk400-k5"):
        chunk_size, k, _ = parse_setting(s)
        assert (chunk_size, k) == (400, 5), f"{s} -> {(chunk_size, k)}"


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
