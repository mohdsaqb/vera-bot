"""POST /v1/context: versioned ingestion over HTTP.

Shapes asserted here come from challenge-testing-brief.md §2.1 and
examples/api-call-examples.md §1.3-1.6.
"""

from __future__ import annotations

from conftest import context_push

MERCHANT_ID = "m_001_drmeera_dentist_delhi"


def test_first_version_is_stored_and_acked(client):
    response = client.post(
        "/v1/context", json=context_push(scope="category", context_id="dentists")
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["ack_id"] == "ack_dentists_v1"
    assert body["stored_at"].endswith("Z")
    assert set(body) == {"accepted", "ack_id", "stored_at"}


def test_stored_payload_is_kept_verbatim(client, store):
    payload = {
        "merchant_id": MERCHANT_ID,
        "category_slug": "dentists",
        "performance": {"views": 2410, "ctr": 0.021},
        "signals": ["stale_posts:22d"],
    }
    client.post("/v1/context", json=context_push(payload=payload))

    assert store.get_payload("merchant", MERCHANT_ID) == payload


def test_same_version_is_a_noop_and_reports_conflict(client, store):
    original = {"merchant_id": MERCHANT_ID, "performance": {"views": 2410}}
    resent = {"merchant_id": MERCHANT_ID, "performance": {"views": 9999}}

    client.post("/v1/context", json=context_push(version=1, payload=original))
    response = client.post("/v1/context", json=context_push(version=1, payload=resent))

    assert response.status_code == 409
    assert response.json() == {
        "accepted": False,
        "reason": "stale_version",
        "current_version": 1,
    }
    # Idempotent: the re-push changed nothing.
    assert store.get_payload("merchant", MERCHANT_ID) == original
    assert len(store.list_scope("merchant")) == 1


def test_same_version_with_different_content_is_a_conflict(client, store):
    """Two different payloads cannot both be version 1; nothing is overwritten."""
    client.post("/v1/context", json=context_push(version=1, payload={"views": 2410}))

    response = client.post("/v1/context", json=context_push(version=1, payload={"views": 1}))

    assert response.status_code == 409
    assert response.json()["reason"] == "stale_version"
    assert store.get_payload("merchant", MERCHANT_ID) == {"views": 2410}


def test_higher_version_replaces_the_stored_context(client, store):
    client.post(
        "/v1/context",
        json=context_push(version=1, payload={"performance": {"views": 2410}}),
    )
    response = client.post(
        "/v1/context",
        json=context_push(version=2, payload={"performance": {"views": 2580}}),
    )

    assert response.status_code == 200
    assert response.json()["ack_id"] == f"ack_{MERCHANT_ID}_v2"
    record = store.get("merchant", MERCHANT_ID)
    assert record is not None
    assert record.version == 2
    assert record.payload == {"performance": {"views": 2580}}


def test_lower_version_never_overwrites_newer_context(client, store):
    client.post(
        "/v1/context",
        json=context_push(version=5, payload={"performance": {"views": 2580}}),
    )
    response = client.post(
        "/v1/context",
        json=context_push(version=3, payload={"performance": {"views": 1}}),
    )

    assert response.status_code == 409
    assert response.json() == {
        "accepted": False,
        "reason": "stale_version",
        "current_version": 5,
    }
    record = store.get("merchant", MERCHANT_ID)
    assert record.version == 5
    assert record.payload == {"performance": {"views": 2580}}


def test_same_context_id_in_different_scopes_stays_isolated(client, store):
    shared_id = "shared_id"
    client.post(
        "/v1/context",
        json=context_push(scope="merchant", context_id=shared_id, payload={"kind": "m"}),
    )
    client.post(
        "/v1/context",
        json=context_push(scope="trigger", context_id=shared_id, version=7,
                          payload={"kind": "t"}),
    )

    assert store.get_payload("merchant", shared_id) == {"kind": "m"}
    assert store.get_payload("trigger", shared_id) == {"kind": "t"}
    assert store.get_version("merchant", shared_id) == 1
    assert store.get_version("trigger", shared_id) == 7


def test_pushes_do_not_mutate_unrelated_contexts(client, store):
    client.post("/v1/context", json=context_push(context_id="m_001", payload={"n": 1}))
    client.post("/v1/context", json=context_push(context_id="m_002", payload={"n": 2}))
    client.post(
        "/v1/context",
        json=context_push(context_id="m_001", version=2, payload={"n": 11}),
    )

    assert store.get_payload("merchant", "m_001") == {"n": 11}
    assert store.get_payload("merchant", "m_002") == {"n": 2}
    assert store.get_version("merchant", "m_002") == 1


def test_repeated_identical_push_is_safe_to_call_many_times(client, store):
    """A retry of the same content reads as success, and changes nothing.

    The judge's warmup treats any `accepted: false` as a failed warmup, so a
    retried push must not look like a rejection; the store still holds one record
    at one version.
    """
    push = context_push(version=4, payload={"n": 1})
    responses = [client.post("/v1/context", json=push) for _ in range(5)]

    assert [r.status_code for r in responses] == [200] * 5
    assert all(r.json()["accepted"] is True for r in responses)
    # The ack is stable, so a retry cannot look like a second distinct push.
    assert len({r.json()["ack_id"] for r in responses}) == 1
    assert len(store.list_scope("merchant")) == 1
    assert store.get_version("merchant", MERCHANT_ID) == 4
    assert store.get_payload("merchant", MERCHANT_ID) == {"n": 1}


def test_unknown_scope_is_rejected_as_invalid_scope(client, store):
    response = client.post("/v1/context", json=context_push(scope="merchants"))

    assert response.status_code == 400
    body = response.json()
    assert body["accepted"] is False
    assert body["reason"] == "invalid_scope"
    assert body["details"]
    assert len(store) == 0


def test_missing_required_fields_are_rejected(client):
    response = client.post("/v1/context", json={"scope": "merchant", "version": 1})

    assert response.status_code == 400
    body = response.json()
    assert body["accepted"] is False
    assert body["reason"] == "invalid_request"


def test_wrong_payload_type_is_rejected(client, store):
    response = client.post(
        "/v1/context",
        json={"scope": "merchant", "context_id": "m_001", "version": 1,
              "payload": "not-an-object"},
    )

    assert response.status_code == 400
    assert response.json()["accepted"] is False
    assert len(store) == 0


def test_negative_version_and_blank_id_are_rejected(client):
    assert client.post("/v1/context", json=context_push(version=-1)).status_code == 400
    assert client.post("/v1/context", json=context_push(context_id="")).status_code == 400


def test_malformed_json_body_is_rejected_without_crashing(client):
    response = client.post(
        "/v1/context",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["accepted"] is False


def test_validation_error_details_never_echo_the_payload(client):
    response = client.post(
        "/v1/context",
        json={"scope": "merchant", "context_id": "m_001", "version": 1,
              "payload": "secret-patient-name"},
    )

    assert "secret-patient-name" not in response.text


def test_delivered_at_is_optional(client, store):
    body = context_push(delivered_at=None)
    assert "delivered_at" not in body

    assert client.post("/v1/context", json=body).status_code == 200
    assert store.get("merchant", MERCHANT_ID).delivered_at is None


def test_extra_envelope_fields_are_tolerated(client):
    body = context_push()
    body["future_judge_field"] = {"anything": True}

    assert client.post("/v1/context", json=body).status_code == 200


def test_teardown_wipes_stored_context(client, store):
    client.post("/v1/context", json=context_push(scope="category", context_id="dentists"))
    client.post("/v1/context", json=context_push())

    response = client.post("/v1/teardown")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "cleared": 2}
    assert len(store) == 0
    assert client.get("/v1/healthz").json()["contexts_loaded"]["merchant"] == 0
