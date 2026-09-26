"""POST /v1/tick and POST /v1/reply: request handling and operational floors.

`/v1/tick` is now backed by the decision engine (see `test_tick_decisions.py`
for the decision behaviour); what is pinned here is the contract floor the judge
depends on: a valid response with no context loaded, tolerance of unknown
trigger ids, rejection of a malformed body, and a `/v1/reply` that can never
emit an empty `send`.

`/v1/reply` now decides for itself (see `test_reply_intents.py` for the intent
and conversation behaviour); what is pinned here is that its wire shape stays
contract-valid whatever it decides.
"""

from __future__ import annotations

from conftest import context_push

TICK_BODY = {
    "now": "2026-04-26T10:35:00Z",
    "available_triggers": ["trg_001_research_digest_dentists"],
}
REPLY_BODY = {
    "conversation_id": "conv_m_001_drmeera_research_W17",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "customer_id": None,
    "from_role": "merchant",
    "message": "Yes please send the abstract.",
    "received_at": "2026-04-26T10:42:00Z",
    "turn_number": 2,
}


def test_tick_returns_an_empty_action_list_without_context(client):
    """With nothing pushed, there is nothing to say and the tick still answers."""
    response = client.post("/v1/tick", json=TICK_BODY)

    assert response.status_code == 200
    assert response.json() == {"actions": []}


def test_tick_works_without_available_triggers(client):
    response = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z"})

    assert response.status_code == 200
    assert response.json() == {"actions": []}


def test_tick_does_not_crash_with_unknown_trigger_ids(client):
    response = client.post(
        "/v1/tick",
        json={"now": "2026-04-26T10:35:00Z", "available_triggers": ["nope", "also_nope"]},
    )

    assert response.status_code == 200
    assert response.json() == {"actions": []}


def test_tick_rejects_a_body_without_now(client):
    response = client.post("/v1/tick", json={"available_triggers": []})

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_tick_does_not_consume_or_alter_context(client, store):
    client.post("/v1/context", json=context_push())
    client.post("/v1/tick", json=TICK_BODY)

    assert store.get_version("merchant", "m_001_drmeera_dentist_delhi") == 1
    assert len(store) == 1


def test_reply_returns_a_contract_valid_action(client):
    """Whatever it decides, the wire shape must hold: a `send` carries a body and
    a CTA, a `wait` carries seconds and no body, an `end` carries neither."""
    response = client.post("/v1/reply", json=REPLY_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["action"] in {"send", "wait", "end"}
    assert body["rationale"]

    if body["action"] == "send":
        assert body["body"].strip()
        assert body["cta"]
        assert "wait_seconds" not in body
    elif body["action"] == "wait":
        assert body["wait_seconds"] > 0
        assert "body" not in body and "cta" not in body
    else:
        assert "body" not in body and "wait_seconds" not in body


def test_reply_to_an_unknown_conversation_does_not_error(client):
    """The replay scenarios post into conversation ids the bot never opened."""
    response = client.post(
        "/v1/reply",
        json={"conversation_id": "conv_never_seen", "merchant_id": "m_999",
              "from_role": "merchant", "message": "Ok lets do it. Whats next?",
              "received_at": "2026-04-26T10:42:00Z", "turn_number": 2},
    )

    assert response.status_code == 200
    assert response.json()["action"] in {"send", "wait", "end"}


def test_reply_accepts_a_customer_turn(client):
    response = client.post(
        "/v1/reply",
        json={
            "conversation_id": "conv_priya_recall",
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "customer_id": "c_001_priya_for_m001",
            "from_role": "customer",
            "message": "1",
            "received_at": "2026-04-26T11:10:00Z",
            "turn_number": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["action"] in {"send", "wait", "end"}


def test_reply_accepts_the_minimal_documented_body(client):
    # api-call-examples.md §2.5 omits merchant_id/customer_id.
    response = client.post(
        "/v1/reply",
        json={
            "conversation_id": "conv_001",
            "from_role": "merchant",
            "message": "Thank you for contacting us! Our team will respond shortly.",
            "received_at": "2026-04-26T10:42:00Z",
            "turn_number": 2,
        },
    )

    assert response.status_code == 200


def test_reply_rejects_a_body_without_conversation_id(client):
    response = client.post(
        "/v1/reply", json={"from_role": "merchant", "message": "hi"}
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_reply_never_returns_an_empty_send_body(client):
    for message in ["", "Not interested. Stop messaging me.", "Ok let's do it"]:
        body = dict(REPLY_BODY, message=message)
        payload = client.post("/v1/reply", json=body).json()

        assert payload["action"] in {"send", "wait", "end"}
        if payload["action"] == "send":
            assert payload["body"].strip()
