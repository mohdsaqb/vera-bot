"""`/v1/tick` and the decision service: resolution, ranking, sending discipline.

The engine decides what to say; this layer decides how much goes out in one tick
and what must not be repeated. Tests drive it through the HTTP surface as the
judge does, plus the service directly for the ordering rules.
"""

from __future__ import annotations

from conftest import NOW, deep_set, variant
from services.decision_service import MAX_ACTIONS_PER_TICK, DecisionService
from state import get_context_store, get_suppression_ledger

REQUIRED_ACTION_FIELDS = {
    "conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id",
    "template_name", "template_params", "body", "cta", "suppression_key", "rationale",
}


def _push(client, scope: str, context_id: str, payload: dict, version: int = 1) -> None:
    """Push one context, tolerating the 409 a repeated same-version push returns.

    Several triggers share a merchant, so `_load` re-pushes the same contexts;
    re-posting a version already held is the documented idempotent no-op.
    """
    response = client.post(
        "/v1/context",
        json={"scope": scope, "context_id": context_id, "version": version,
              "payload": payload, "delivered_at": NOW},
    )
    assert response.status_code in {200, 409}, response.text


def _load(client, dataset, trigger_ids: list[str]) -> None:
    """Push the contexts the given triggers depend on, as the judge would."""
    for trigger_id in trigger_ids:
        trigger = dataset.triggers[trigger_id]
        merchant = dataset.merchants[trigger["merchant_id"]]
        _push(client, "merchant", merchant["merchant_id"], merchant)
        _push(client, "category", merchant["category_slug"],
              dataset.categories[merchant["category_slug"]])
        if trigger.get("customer_id"):
            customer = dataset.customers[trigger["customer_id"]]
            _push(client, "customer", customer["customer_id"], customer)
        _push(client, "trigger", trigger_id, trigger)


def _tick(client, trigger_ids: list[str], now: str = NOW) -> list[dict]:
    response = client.post("/v1/tick", json={"now": now, "available_triggers": trigger_ids})
    assert response.status_code == 200, response.text
    return response.json()["actions"]


class TestTickEndpoint:
    def test_no_context_means_no_actions(self, client):
        assert _tick(client, ["trg_001_research_digest_dentists"]) == []

    def test_unknown_trigger_ids_are_skipped(self, client, dataset):
        _load(client, dataset, ["trg_001_research_digest_dentists"])
        actions = _tick(client, ["nope", "trg_001_research_digest_dentists", "also_nope"])

        assert len(actions) == 1
        assert actions[0]["trigger_id"] == "trg_001_research_digest_dentists"

    def test_action_carries_every_required_field(self, client, dataset):
        _load(client, dataset, ["trg_001_research_digest_dentists"])
        action = _tick(client, ["trg_001_research_digest_dentists"])[0]

        assert set(action) == REQUIRED_ACTION_FIELDS
        assert action["body"] and action["rationale"] and action["suppression_key"]
        assert action["send_as"] == "vera"
        assert action["merchant_id"] == "m_001_drmeera_dentist_delhi"

    def test_a_customer_scoped_trigger_yields_a_merchant_on_behalf_action(self, client, dataset):
        _load(client, dataset, ["trg_003_recall_due_priya"])
        action = _tick(client, ["trg_003_recall_due_priya"])[0]

        assert action["send_as"] == "merchant_on_behalf"
        assert action["customer_id"] == "c_001_priya_for_m001"

    def test_tick_without_triggers_is_still_valid(self, client, dataset):
        _load(client, dataset, ["trg_001_research_digest_dentists"])
        assert _tick(client, []) == []

    def test_empty_actions_when_every_trigger_is_expired(self, client, dataset):
        _load(client, dataset, ["trg_010_ipl_match_delhi"])
        assert _tick(client, ["trg_010_ipl_match_delhi"], now="2026-06-01T00:00:00Z") == []


class TestSendingDiscipline:
    def test_one_action_per_merchant_per_tick(self, client, dataset):
        """Several stories can qualify at once; the merchant gets the strongest."""
        ids = ["trg_001_research_digest_dentists", "trg_002_compliance_dci_radiograph",
               "trg_022_cde_webinar_dentists", "trg_023_competitor_opened_dentist"]
        _load(client, dataset, ids)
        actions = _tick(client, ids)

        assert len(actions) == 1
        # The compliance item carries urgency 4 against the others' 1-2.
        assert actions[0]["trigger_id"] == "trg_002_compliance_dci_radiograph"

    def test_different_merchants_each_get_one(self, client, dataset):
        ids = ["trg_001_research_digest_dentists", "trg_004_perf_dip_bharat",
               "trg_010_ipl_match_delhi"]
        _load(client, dataset, ids)
        actions = _tick(client, ids)

        assert len(actions) == 3
        assert len({action["merchant_id"] for action in actions}) == 3

    def test_a_story_is_not_told_twice_across_ticks(self, client, dataset):
        _load(client, dataset, ["trg_001_research_digest_dentists"])

        first = _tick(client, ["trg_001_research_digest_dentists"])
        second = _tick(client, ["trg_001_research_digest_dentists"])

        assert len(first) == 1
        assert second == [], "the same suppression key went out twice"

    def test_the_ledger_records_the_key_that_was_used(self, client, dataset, ledger):
        _load(client, dataset, ["trg_001_research_digest_dentists"])
        _tick(client, ["trg_001_research_digest_dentists"])

        assert ledger.is_suppressed("research:dentists:2026-W17")
        assert not ledger.is_suppressed("research:dentists:2026-W18")

    def test_a_later_tick_can_pick_up_the_next_story(self, client, dataset):
        ids = ["trg_001_research_digest_dentists", "trg_002_compliance_dci_radiograph"]
        _load(client, dataset, ids)

        first = _tick(client, ids)
        second = _tick(client, ids)

        assert len(first) == 1 and len(second) == 1
        assert first[0]["trigger_id"] != second[0]["trigger_id"]

    def test_the_action_cap_is_respected(self, client, dataset):
        ids = list(dataset.triggers)
        _load(client, dataset, ids)
        actions = _tick(client, ids)

        assert len(actions) <= MAX_ACTIONS_PER_TICK
        assert len({action["merchant_id"] for action in actions}) == len(actions)

    def test_bodies_are_never_repeated_within_a_conversation(self, client, dataset, ledger):
        _load(client, dataset, ["trg_004_perf_dip_bharat"])
        action = _tick(client, ["trg_004_perf_dip_bharat"])[0]

        assert ledger.is_repeat(action["conversation_id"], action["body"])
        assert not ledger.is_repeat(action["conversation_id"], "something else entirely")


class TestContextUpdates:
    def test_a_new_context_version_is_used_on_the_next_tick(self, client, dataset):
        """The judge injects fresh context mid-test; a later send must use it."""
        merchant = dataset.merchants["m_002_bharat_dentist_mumbai"]
        _load(client, dataset, ["trg_004_perf_dip_bharat"])
        first = _tick(client, ["trg_004_perf_dip_bharat"])[0]
        assert "1.8%" in first["body"]

        # Version 2 of the merchant: a different click-through rate.
        updated = deep_set(merchant, "performance.ctr", 0.010)
        _push(client, "merchant", merchant["merchant_id"], updated, version=2)

        # A new week's trigger, so suppression does not mask the change.
        next_week = variant(dataset.triggers["trg_004_perf_dip_bharat"],
                            id="trg_004b", suppression_key="perf_dip:m_002:calls:2026-W18")
        _push(client, "trigger", "trg_004b", next_week)
        second = _tick(client, ["trg_004b"])[0]

        assert "1%" in second["body"]
        assert second["body"] != first["body"]

    def test_stale_context_pushes_do_not_change_what_is_sent(self, client, dataset):
        merchant = dataset.merchants["m_002_bharat_dentist_mumbai"]
        _load(client, dataset, ["trg_004_perf_dip_bharat"])
        _push(client, "merchant", merchant["merchant_id"],
              deep_set(merchant, "performance.ctr", 0.010), version=5)

        stale = client.post("/v1/context", json={
            "scope": "merchant", "context_id": merchant["merchant_id"], "version": 2,
            "payload": deep_set(merchant, "performance.ctr", 0.099), "delivered_at": NOW,
        })
        assert stale.status_code == 409

        action = _tick(client, ["trg_004_perf_dip_bharat"])[0]
        assert "9.9%" not in action["body"]


class TestDecisionService:
    def test_evaluation_is_ranked_best_first(self, dataset):
        store = get_context_store()
        for trigger_id in ("trg_012_milestone_mylari", "trg_018_supply_atorvastatin_recall"):
            trigger = dataset.triggers[trigger_id]
            merchant = dataset.merchants[trigger["merchant_id"]]
            store.save("trigger", trigger_id, 1, trigger)
            store.save("merchant", merchant["merchant_id"], 1, merchant)
            store.save("category", merchant["category_slug"], 1,
                       dataset.categories[merchant["category_slug"]])

        service = DecisionService(store, get_suppression_ledger())
        ranked = service.evaluate_available(
            ["trg_012_milestone_mylari", "trg_018_supply_atorvastatin_recall"], NOW
        )

        assert [evaluation.trigger_id for evaluation, _ in ranked][0] == (
            "trg_018_supply_atorvastatin_recall"
        )

    def test_teardown_clears_the_suppression_ledger(self, client, dataset, ledger):
        _load(client, dataset, ["trg_001_research_digest_dentists"])
        assert _tick(client, ["trg_001_research_digest_dentists"])
        assert len(ledger) == 1

        assert client.post("/v1/teardown").status_code == 200
        assert len(ledger) == 0
