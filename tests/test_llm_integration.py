"""The writer inside the real pipeline: `compose()`, `/v1/tick`, and determinism.

Two things are pinned here. First, that generation only ever changes wording: the trigger, signal, offer, CTA, sender, suppression key and rationale come out
of a generated message exactly as they came out of a deterministic one. Second,
that the API keeps working when the model does not.
"""

from __future__ import annotations

from conftest import NOW, FakeWriter
from engine import compose
from engine.types import GenerationBrief
from services.decision_service import DecisionService
from services.llm_service import TickBudget
from state import get_context_store, get_conversation_store, get_suppression_ledger

# A faithful rewording of the perf-dip brief: same facts, same offer, same one ask.
GOOD_WORDING = (
    "Dr. Bharat — calls are down 50% over the last 7 days against a baseline of 12, "
    "and your click-through rate sits at 1.8% against the 3% median. "
    "Want me to put Aligner Consultation @ ₹499 live?"
)


def _args(dataset, trigger_id="trg_004_perf_dip_bharat"):
    return dataset.compose_args(trigger_id)


# --------------------------------------------------------------------------- #
# Wording is the only thing generation may change
# --------------------------------------------------------------------------- #
class TestDecisionIsUntouched:
    def test_a_generated_message_keeps_every_deterministic_field(self, dataset):
        category, merchant, trigger, customer = _args(dataset)

        baseline = compose(category, merchant, trigger, customer, now=NOW)
        generated = compose(
            category, merchant, trigger, customer, now=NOW,
            writer=FakeWriter(bodies=[GOOD_WORDING]),
        )

        assert generated.body != baseline.body, "the wording should have changed"
        assert generated.body_source == "llm"
        # Everything the engine decided is identical.
        assert generated.cta == baseline.cta
        assert generated.send_as == baseline.send_as
        assert generated.suppression_key == baseline.suppression_key
        assert generated.rationale == baseline.rationale
        assert generated.conversation_id == baseline.conversation_id
        assert generated.trigger_id == baseline.trigger_id
        assert generated.template_name == baseline.template_name
        assert generated.plan.action.key == baseline.plan.action.key
        assert generated.plan.primary_fact == baseline.plan.primary_fact
        assert generated.plan.offer == baseline.plan.offer

    def test_a_writer_cannot_turn_a_no_action_into_a_send(self, dataset):
        """The festival trigger is 188 days out; no wording makes that worth sending."""
        category, merchant, trigger, customer = _args(dataset, "trg_006_festival_diwali")

        result = compose(
            category, merchant, trigger, customer, now=NOW,
            writer=FakeWriter(bodies=["Diwali is coming, want a campaign?"]),
        )

        assert result.decision == "no_action"
        assert result.body == ""

    def test_a_writer_is_never_called_for_a_no_action(self, dataset):
        category, merchant, trigger, customer = _args(dataset, "trg_006_festival_diwali")
        writer = FakeWriter(bodies=["anything"])

        compose(category, merchant, trigger, customer, now=NOW, writer=writer)

        assert writer.calls == 0

    def test_the_brief_hides_the_decision_machinery(self, dataset):
        category, merchant, trigger, customer = _args(dataset)
        writer = FakeWriter()

        compose(category, merchant, trigger, customer, now=NOW, writer=writer)

        brief: GenerationBrief = writer.seen[0]
        assert brief.facts
        assert brief.ask
        assert brief.deterministic_body
        # None of these belong to a writer.
        assert not hasattr(brief, "suppression_key")
        assert not hasattr(brief, "priority")
        assert not hasattr(brief, "evidence")
        rendered = str(brief)
        for leak in ("perf_dip:m_002", "trigger.payload", "merchant.performance"):
            assert leak not in rendered


# --------------------------------------------------------------------------- #
# Fallbacks
# --------------------------------------------------------------------------- #
class TestFallback:
    def test_an_ungrounded_generation_never_reaches_the_output(self, dataset):
        category, merchant, trigger, customer = _args(dataset)
        writer = FakeWriter(
            bodies=["Dr. Bharat — calls fell 82%. Want me to run a ₹1,999 implant offer?"]
        )

        result = compose(category, merchant, trigger, customer, now=NOW, writer=writer)

        assert result.body_source == "deterministic"
        assert "82%" not in result.body
        assert "₹1,999" not in result.body

    def test_a_raising_writer_still_produces_a_message(self, dataset):
        category, merchant, trigger, customer = _args(dataset)

        result = compose(
            category, merchant, trigger, customer, now=NOW,
            writer=FakeWriter(raises=TimeoutError("timed out")),
        )

        assert result.should_send
        assert result.body_source == "deterministic"
        assert result.body

    def test_an_empty_generation_falls_back(self, dataset):
        category, merchant, trigger, customer = _args(dataset)

        result = compose(
            category, merchant, trigger, customer, now=NOW, writer=FakeWriter(bodies=[""])
        )

        assert result.should_send
        assert result.body_source == "deterministic"

    def test_the_fallback_is_byte_identical_to_the_no_writer_output(self, dataset):
        category, merchant, trigger, customer = _args(dataset)

        plain = compose(category, merchant, trigger, customer, now=NOW)
        fell_back = compose(
            category, merchant, trigger, customer, now=NOW,
            writer=FakeWriter(raises=RuntimeError("boom")),
        )

        assert fell_back.body == plain.body
        assert fell_back.to_dict() == plain.to_dict()


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_the_deterministic_path_is_unchanged_by_phase_three(self, dataset):
        """Same inputs, no writer: byte-identical across calls."""
        for trigger_id in dataset.triggers:
            category, merchant, trigger, customer = dataset.compose_args(trigger_id)
            first = compose(category, merchant, trigger, customer, now=NOW)
            second = compose(category, merchant, trigger, customer, now=NOW)

            assert first.to_dict() == second.to_dict()
            assert first.body_source == "deterministic"

    def test_decisions_are_identical_with_and_without_a_writer(self, dataset):
        for trigger_id in dataset.triggers:
            category, merchant, trigger, customer = dataset.compose_args(trigger_id)
            plain = compose(category, merchant, trigger, customer, now=NOW)
            with_writer = compose(
                category, merchant, trigger, customer, now=NOW,
                writer=FakeWriter(bodies=[GOOD_WORDING]),
            )

            assert with_writer.decision == plain.decision
            assert with_writer.reason_code == plain.reason_code
            assert with_writer.cta == plain.cta
            assert with_writer.send_as == plain.send_as
            assert with_writer.suppression_key == plain.suppression_key
            assert with_writer.rationale == plain.rationale

    def test_the_same_brief_is_produced_every_time(self, dataset):
        category, merchant, trigger, customer = _args(dataset)
        first, second = FakeWriter(), FakeWriter()

        compose(category, merchant, trigger, customer, now=NOW, writer=first)
        compose(category, merchant, trigger, customer, now=NOW, writer=second)

        assert first.seen[0] == second.seen[0]


# --------------------------------------------------------------------------- #
# /v1/tick
# --------------------------------------------------------------------------- #
class TestTickIntegration:
    def _load(self, store, dataset, trigger_ids):
        for trigger_id in trigger_ids:
            trigger = dataset.triggers[trigger_id]
            merchant = dataset.merchants[trigger["merchant_id"]]
            store.save("trigger", trigger_id, 1, trigger)
            store.save("merchant", merchant["merchant_id"], 1, merchant)
            store.save("category", merchant["category_slug"], 1,
                       dataset.categories[merchant["category_slug"]])

    def test_tick_uses_the_writer_and_opens_conversations(self, dataset):
        store, conversations = get_context_store(), get_conversation_store()
        self._load(store, dataset, ["trg_004_perf_dip_bharat"])
        writer = FakeWriter(bodies=[GOOD_WORDING])
        service = DecisionService(store, get_suppression_ledger(), conversations, writer)

        actions = service.actions(["trg_004_perf_dip_bharat"], NOW)

        assert len(actions) == 1
        assert actions[0]["body"] == GOOD_WORDING
        state = conversations.get(actions[0]["conversation_id"])
        assert state is not None
        assert state.merchant_id == "m_002_bharat_dentist_mumbai"
        assert state.brief is not None
        assert state.last_outbound.body == GOOD_WORDING

    def test_a_failing_writer_does_not_fail_the_tick(self, dataset):
        store = get_context_store()
        self._load(store, dataset, ["trg_004_perf_dip_bharat", "trg_001_research_digest_dentists"])
        service = DecisionService(
            store, get_suppression_ledger(), get_conversation_store(),
            FakeWriter(raises=ConnectionError("provider down")),
        )

        actions = service.actions(
            ["trg_004_perf_dip_bharat", "trg_001_research_digest_dentists"], NOW
        )

        assert len(actions) == 2
        assert all(action["body"].strip() for action in actions)

    def test_the_tick_budget_caps_generation(self, dataset):
        store = get_context_store()
        trigger_ids = list(dataset.triggers)
        self._load(store, dataset, trigger_ids)

        class Budgeted(FakeWriter):
            def with_budget(self, budget):
                self.budget = budget
                return self

        writer = Budgeted(bodies=[GOOD_WORDING])
        service = DecisionService(store, get_suppression_ledger(), get_conversation_store(), writer)

        service.actions(trigger_ids, NOW)

        assert isinstance(writer.budget, TickBudget)
        assert writer.budget.max_calls > 0

    def test_an_opted_out_merchant_gets_no_further_action(self, dataset):
        store, conversations = get_context_store(), get_conversation_store()
        ledger = get_suppression_ledger()
        self._load(store, dataset, ["trg_004_perf_dip_bharat"])
        service = DecisionService(store, ledger, conversations, None)

        first = service.actions(["trg_004_perf_dip_bharat"], NOW)
        assert len(first) == 1

        from services.reply_service import ReplyService

        ReplyService(conversations, ledger).handle(
            first[0]["conversation_id"], "Stop messaging me.",
            merchant_id="m_002_bharat_dentist_mumbai",
        )
        store.save("trigger", "trg_005_renewal_due_bharat", 1,
                   dataset.triggers["trg_005_renewal_due_bharat"])

        assert service.actions(["trg_005_renewal_due_bharat"], NOW) == []

    def test_tick_over_http_with_a_writer_configured(self, client, dataset, monkeypatch):
        writer = FakeWriter(bodies=[GOOD_WORDING])
        monkeypatch.setattr("state._writer", writer)
        trigger = dataset.triggers["trg_004_perf_dip_bharat"]
        merchant = dataset.merchants[trigger["merchant_id"]]
        for scope, context_id, payload in [
            ("category", merchant["category_slug"], dataset.categories[merchant["category_slug"]]),
            ("merchant", merchant["merchant_id"], merchant),
            ("trigger", trigger["id"], trigger),
        ]:
            client.post("/v1/context", json={
                "scope": scope, "context_id": context_id, "version": 1,
                "payload": payload, "delivered_at": NOW,
            })

        response = client.post(
            "/v1/tick", json={"now": NOW, "available_triggers": [trigger["id"]]}
        )

        assert response.status_code == 200
        actions = response.json()["actions"]
        assert len(actions) == 1
        assert actions[0]["body"] == GOOD_WORDING


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
class TestSecrets:
    def test_settings_carry_no_key(self):
        from config import get_settings

        assert "GROQ_API_KEY" not in str(get_settings())
        assert not any("key" in field for field in vars(get_settings()))

    def test_metadata_never_exposes_a_key(self, client, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-super-secret-value")

        payload = client.get("/v1/metadata").json()

        assert "gsk-super-secret-value" not in str(payload)
        assert not any("key" in str(key).lower() for key in payload)

    def test_healthz_never_exposes_a_key(self, client, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-super-secret-value")

        assert "gsk-super-secret-value" not in str(client.get("/v1/healthz").json())

    def test_the_prompt_never_carries_a_key(self, brief, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-super-secret-value")
        from services.llm_service import render_prompt

        assert "gsk-super-secret-value" not in render_prompt(brief)
