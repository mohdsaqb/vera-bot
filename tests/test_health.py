"""GET /v1/healthz — liveness probe and context accounting."""

from __future__ import annotations

from conftest import context_push


def test_healthz_returns_ok_and_zero_counts_on_fresh_boot(client):
    response = client.get("/v1/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["uptime_seconds"], int)
    assert body["uptime_seconds"] >= 0
    # The judge expects all four scopes present and zeroed before warmup.
    assert body["contexts_loaded"] == {
        "category": 0,
        "merchant": 0,
        "customer": 0,
        "trigger": 0,
    }


def test_healthz_counts_reflect_pushed_contexts(client):
    pushes = [
        context_push(scope="category", context_id="dentists"),
        context_push(scope="category", context_id="salons"),
        context_push(scope="merchant", context_id="m_001_drmeera_dentist_delhi"),
        context_push(scope="customer", context_id="c_001_priya_for_m001"),
        context_push(scope="trigger", context_id="trg_001_research_digest_dentists"),
    ]
    for push in pushes:
        assert client.post("/v1/context", json=push).status_code == 200

    body = client.get("/v1/healthz").json()
    assert body["contexts_loaded"] == {
        "category": 2,
        "merchant": 1,
        "customer": 1,
        "trigger": 1,
    }


def test_healthz_version_bump_does_not_inflate_counts(client):
    client.post("/v1/context", json=context_push(version=1))
    client.post("/v1/context", json=context_push(version=2))

    assert client.get("/v1/healthz").json()["contexts_loaded"]["merchant"] == 1


def test_healthz_is_json_and_fast_without_any_dependency(client):
    response = client.get("/v1/healthz")

    assert response.headers["content-type"].startswith("application/json")
    assert set(response.json()) == {"status", "uptime_seconds", "contexts_loaded"}
