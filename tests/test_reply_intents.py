"""`/v1/reply` behaviour: what each intent does to the thread, and the state it leaves.

Driven through `ReplyService` (the unit the endpoint delegates to) and through
HTTP where the wire shape matters. The three replay scenarios from
`examples/api-call-examples.md` §4 are pinned end to end.
"""

from __future__ import annotations

import pytest

from conftest import NOW, FakeWriter
from engine.intent import (
    ACCEPT,
    AMBIGUOUS,
    AUTO_REPLY,
    DELAY,
    HOSTILE,
    OFF_TOPIC,
    QUESTION,
    REJECT,
)
from engine.reply import (
    QUALIFYING_PHRASES,
    WAIT_AFTER_DELAY_SOON,
    WAIT_AFTER_DELAY_TOMORROW,
    WAIT_AFTER_SECOND_AUTO_REPLY,
    re_qualifies,
)
from engine.types import GenerationBrief
from services.conversation_store import ConversationStore
from services.reply_service import ReplyService
from services.suppression import SuppressionLedger

AUTO_REPLY_TEXT = "Thank you for contacting us! Our team will respond shortly."


def make_brief(**overrides) -> GenerationBrief:
    defaults = {
        "purpose": "outbound",
        "category": "Dentists",
        "audience": "merchant",
        "recipient_name": "Meera",
        "salutation": "Dr. Meera",
        "tone": "peer_clinical",
        "language": "en",
        "reason": "perf dip",
        "facts": ("calls are down 50% week-on-week (18 in the last 30 days)",),
        "offer_title": "Dental Cleaning @ ₹299",
        "offer_is_live": True,
        "ask": "Want me to put Dental Cleaning @ ₹299 live?",
        "cta": "binary_yes_no",
        "deterministic_body": (
            "Dr. Meera — calls are down 50% week-on-week. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        ),
    }
    defaults.update(overrides)
    return GenerationBrief(**defaults)


@pytest.fixture
def conversations() -> ConversationStore:
    return ConversationStore()


@pytest.fixture
def service(conversations) -> ReplyService:
    return ReplyService(conversations, SuppressionLedger())


@pytest.fixture
def open_thread(conversations):
    """A thread the bot opened, the way `/v1/tick` would."""

    def _open(conversation_id="conv_1", merchant_id="m_001", **brief_overrides):
        brief = make_brief(**brief_overrides)
        conversations.open_from_action(
            conversation_id,
            merchant_id=merchant_id,
            customer_id=None,
            trigger_id="trg_004_perf_dip_bharat",
            family="performance",
            action_key="activate_offer",
            cta=brief.cta,
            send_as="vera",
            audience=brief.audience,
            language=brief.language,
            category_slug="dentists",
            suppression_key="perf_dip:m_001:calls:2026-W17",
            brief=brief,
            body=brief.deterministic_body,
            at=NOW,
        )
        return conversation_id

    return _open


# --------------------------------------------------------------------------- #
# Accept — the intent transition
# --------------------------------------------------------------------------- #
class TestAccept:
    @pytest.mark.parametrize(
        "message", ["yes", "Yes, do it.", "okay", "go ahead", "Ok lets do it. Whats next?"]
    )
    def test_a_commitment_moves_to_the_action(self, service, open_thread, message):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, message, merchant_id="m_001")

        assert outcome.action == "send"
        assert outcome.intent == ACCEPT
        assert outcome.body

    def test_the_action_reply_never_re_qualifies(self, service, open_thread):
        """The named failure in challenge-brief.md §9 pattern D."""
        conversation_id = open_thread()

        body = service.handle(conversation_id, "Yes, do it.").body.lower()

        for phrase in QUALIFYING_PHRASES:
            assert phrase not in body, f"re-qualified with {phrase!r}"
        assert re_qualifies(body) is False

    def test_the_action_reply_names_the_next_step_and_asks_once(self, service, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "Yes, do it.")

        assert "Dental Cleaning @ ₹299" in outcome.body
        assert "CONFIRM" in outcome.body
        assert outcome.cta == "binary_confirm_cancel"
        assert outcome.body.count("?") == 0

    def test_it_does_not_claim_the_work_is_already_done(self, service, open_thread):
        """There is no execution tool behind this bot, so nothing may be claimed."""
        conversation_id = open_thread()

        body = service.handle(conversation_id, "Yes, do it.").body.lower()

        for false_claim in ("done!", "i have activated", "is now live", "has been published",
                            "i've updated", "activated it"):
            assert false_claim not in body

    def test_the_judge_simulator_heuristic_passes(self, service):
        """`judge_simulator.py` greps for action words and against qualifiers."""
        outcome = service.handle("conv_intent_1", "Ok lets do it. Whats next?", merchant_id="m_1")
        body = outcome.body.lower()

        assert any(w in body for w in ("done", "sending", "draft", "here", "confirm",
                                       "proceed", "next"))
        assert not any(w in body for w in ("would you", "do you", "can you tell",
                                          "what if", "how about"))

    def test_an_unknown_thread_still_transitions(self, service):
        outcome = service.handle("conv_never_opened", "yes do it", merchant_id="m_777")

        assert outcome.action == "send"
        assert outcome.intent == ACCEPT


# --------------------------------------------------------------------------- #
# Reject and hostility
# --------------------------------------------------------------------------- #
class TestReject:
    @pytest.mark.parametrize("message", ["no", "not interested", "no thanks", "skip it"])
    def test_a_refusal_closes_the_thread(self, service, conversations, open_thread, message):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, message, merchant_id="m_001")

        assert outcome.action == "end"
        assert outcome.intent == REJECT
        state = conversations.get(conversation_id)
        assert state.active is False
        assert state.should_end is True

    def test_the_same_offer_is_not_pushed_again(self, service, open_thread):
        conversation_id = open_thread()
        service.handle(conversation_id, "not interested")

        again = service.handle(conversation_id, "yes actually")

        assert again.action == "end", "a closed thread stays closed"

    def test_hostility_ends_immediately(self, service, conversations, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(
            conversation_id, "Stop messaging me. This is useless spam.", merchant_id="m_001"
        )

        assert outcome.action == "end"
        assert outcome.intent == HOSTILE
        assert conversations.get(conversation_id).end_reason == "opted_out"

    def test_an_opt_out_suppresses_the_merchant_not_just_the_thread(
        self, conversations, open_thread
    ):
        ledger = SuppressionLedger()
        service = ReplyService(conversations, ledger)
        conversation_id = open_thread()

        service.handle(conversation_id, "Stop messaging me.", merchant_id="m_001")

        assert ledger.is_suppressed("optout:m_001")
        assert conversations.is_merchant_closed("m_001")


# --------------------------------------------------------------------------- #
# Not now
# --------------------------------------------------------------------------- #
class TestDelay:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("not now", WAIT_AFTER_DELAY_SOON),
            ("later", WAIT_AFTER_DELAY_SOON),
            ("tomorrow", WAIT_AFTER_DELAY_TOMORROW),
            ("maybe next week", WAIT_AFTER_DELAY_TOMORROW),
        ],
    )
    def test_a_postponement_backs_off_for_as_long_as_asked(
        self, service, open_thread, message, expected
    ):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, message)

        assert outcome.action == "wait"
        assert outcome.intent == DELAY
        assert outcome.wait_seconds == expected

    def test_a_postponement_is_not_a_rejection(self, service, conversations, open_thread):
        conversation_id = open_thread()

        service.handle(conversation_id, "not now")

        state = conversations.get(conversation_id)
        assert state.active is True
        assert state.should_end is False
        assert state.delay_requested_seconds == WAIT_AFTER_DELAY_SOON

    def test_nothing_is_sent_in_the_same_turn(self, service, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "later")

        assert outcome.body == ""


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #
class TestQuestion:
    def test_a_question_is_answered_from_the_stored_fact(self, service, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "how much?")

        assert outcome.action == "send"
        assert outcome.intent == QUESTION
        assert "50%" in outcome.body

    def test_the_answer_introduces_no_new_fact(self, service, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "what do you mean?")

        for invented in ("₹499", "60%", "next month"):
            assert invented not in outcome.body

    def test_with_no_stored_facts_it_says_so_rather_than_guessing(self, service):
        outcome = service.handle("conv_bare", "how much?", merchant_id="m_001")

        assert outcome.action == "send"
        assert "don't have that detail" in outcome.body.lower()
        assert "check" in outcome.body.lower()


# --------------------------------------------------------------------------- #
# Off topic
# --------------------------------------------------------------------------- #
class TestOffTopic:
    def test_an_out_of_scope_request_is_declined_and_redirected(self, service, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(
            conversation_id, "Btw can you also help me with my GST filing this month?"
        )

        assert outcome.action == "send"
        assert outcome.intent == OFF_TOPIC
        assert "outside what I can help with" in outcome.body
        assert "Dental Cleaning @ ₹299" in outcome.body, "should return to the original ask"

    def test_the_thread_stays_open(self, service, conversations, open_thread):
        conversation_id = open_thread()

        service.handle(conversation_id, "can you help with my GST?")

        assert conversations.get(conversation_id).active is True

    def test_without_a_live_ask_it_invites_something_in_scope(self, service):
        """A thread with no stored ask must not redirect back at the detour."""
        outcome = service.handle("conv_unknown_off", "can you file my GST return?",
                                 merchant_id="m_001")

        assert outcome.action == "send"
        assert "carry on with that" not in outcome.body
        assert "listing or offers" in outcome.body

    def test_it_does_not_attempt_the_out_of_scope_task(self, service, open_thread):
        conversation_id = open_thread()

        body = service.handle(conversation_id, "can you file my GST return?").body.lower()

        assert "gst" not in body, "must not pretend to engage with the detour"


# --------------------------------------------------------------------------- #
# Auto-reply — the replay scenario
# --------------------------------------------------------------------------- #
class TestAutoReply:
    def test_the_first_canned_reply_gets_one_prompt_for_the_owner(self, service, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, AUTO_REPLY_TEXT, merchant_id="m_001")

        assert outcome.action == "send"
        assert outcome.intent == AUTO_REPLY
        assert "auto-responder" in outcome.body
        assert outcome.body.count("?") == 0

    def test_the_second_backs_off_a_day(self, service, open_thread):
        conversation_id = open_thread()
        service.handle(conversation_id, AUTO_REPLY_TEXT, merchant_id="m_001")

        outcome = service.handle(conversation_id, AUTO_REPLY_TEXT, merchant_id="m_001")

        assert outcome.action == "wait"
        assert outcome.wait_seconds == WAIT_AFTER_SECOND_AUTO_REPLY

    def test_the_third_closes_the_thread(self, service, conversations, open_thread):
        conversation_id = open_thread()
        for _ in range(3):
            outcome = service.handle(conversation_id, AUTO_REPLY_TEXT, merchant_id="m_001")

        assert outcome.action == "end"
        assert conversations.get(conversation_id).end_reason == "auto_reply_only"

    def test_four_canned_turns_never_burn_more_than_one_send(self, service, open_thread):
        """Production Vera's documented waste is 2-3 turns per auto-reply thread."""
        conversation_id = open_thread()

        actions = [
            service.handle(conversation_id, AUTO_REPLY_TEXT, merchant_id="m_001").action
            for _ in range(4)
        ]

        assert actions == ["send", "wait", "end", "end"]
        assert actions.count("send") == 1

    def test_canned_replies_are_counted_across_the_merchants_threads(self, service):
        """`judge_simulator.py` rotates the conversation id on every canned turn."""
        actions = [
            service.handle(f"conv_auto_{i}", AUTO_REPLY_TEXT, merchant_id="m_001").action
            for i in range(1, 5)
        ]

        assert "end" in actions, f"never closed across rotating threads: {actions}"
        assert actions.index("end") <= 3

    def test_a_real_reply_after_a_canned_one_resets_the_count(
        self, service, conversations, open_thread
    ):
        conversation_id = open_thread()
        service.handle(conversation_id, AUTO_REPLY_TEXT, merchant_id="m_001")

        outcome = service.handle(conversation_id, "yes go ahead", merchant_id="m_001")

        assert outcome.action == "send"
        assert outcome.intent == ACCEPT
        assert conversations.get(conversation_id).consecutive_auto_replies == 0


# --------------------------------------------------------------------------- #
# Ambiguity and stopping
# --------------------------------------------------------------------------- #
class TestAmbiguous:
    def test_the_first_unclear_reply_gets_one_plain_restatement(self, service, open_thread):
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "hmm")

        assert outcome.action == "send"
        assert outcome.intent == AMBIGUOUS
        assert outcome.body.count("?") <= 1

    def test_a_second_unclear_reply_backs_off(self, service, open_thread):
        conversation_id = open_thread()
        service.handle(conversation_id, "hmm")

        outcome = service.handle(conversation_id, "asdf")

        assert outcome.action == "wait"

    def test_a_thread_is_never_answered_with_a_repeat(self, service, conversations, open_thread):
        conversation_id = open_thread()
        for _ in range(4):
            service.handle(conversation_id, "hmm")

        bodies = [t.body for t in conversations.get(conversation_id).outbound]
        assert len(bodies) == len(set(bodies)), "the same body went out twice"


# --------------------------------------------------------------------------- #
# Conversation state
# --------------------------------------------------------------------------- #
class TestConversationState:
    def test_a_thread_records_both_directions_in_order(self, service, conversations, open_thread):
        conversation_id = open_thread()

        service.handle(conversation_id, "how much?")

        state = conversations.get(conversation_id)
        assert [t.direction for t in state.turns] == ["outbound", "inbound", "outbound"]
        assert state.turn_count == 3
        assert state.last_intent == QUESTION

    def test_the_last_outbound_is_tracked(self, service, conversations, open_thread):
        conversation_id = open_thread()

        service.handle(conversation_id, "yes do it")

        last = conversations.get(conversation_id).last_outbound
        assert last is not None
        assert "CONFIRM" in last.body
        assert last.cta == "binary_confirm_cancel"

    def test_real_engagement_is_distinguished_from_canned(
        self, service, conversations, open_thread
    ):
        canned = open_thread("conv_canned")
        engaged = open_thread("conv_engaged")

        service.handle(canned, AUTO_REPLY_TEXT, merchant_id="m_001")
        service.handle(engaged, "how much?", merchant_id="m_001")

        assert conversations.get(canned).has_real_engagement is False
        assert conversations.get(engaged).has_real_engagement is True

    def test_a_slot_ask_makes_a_bare_number_an_acceptance(
        self, service, conversations, open_thread
    ):
        conversation_id = open_thread(cta="multi_choice_slot")

        outcome = service.handle(conversation_id, "1")

        assert conversations.get(conversation_id).expects_slot is False  # after the reply
        assert outcome.intent == ACCEPT

    def test_state_is_cleared_by_reset(self, conversations, open_thread):
        open_thread()

        assert len(conversations) == 1
        assert conversations.clear() == 1
        assert len(conversations) == 0


# --------------------------------------------------------------------------- #
# Wording layer on the reply path
# --------------------------------------------------------------------------- #
class TestReplyWording:
    def test_a_grounded_generation_is_used(self, conversations, open_thread):
        good = (
            "Right — I'll set Dental Cleaning @ ₹299 up and send it here to approve "
            "first. Reply CONFIRM and I'll proceed."
        )
        writer = FakeWriter(bodies=[good])
        service = ReplyService(conversations, SuppressionLedger(), writer)
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "yes do it")

        assert outcome.body == good
        assert outcome.body_source == "llm"

    def test_an_ungrounded_generation_falls_back_to_the_draft(self, conversations, open_thread):
        writer = FakeWriter(bodies=["Done — I've activated your ₹99 offer already."])
        service = ReplyService(conversations, SuppressionLedger(), writer)
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "yes do it")

        assert outcome.body_source == "deterministic"
        assert "₹99" not in outcome.body

    def test_a_re_qualifying_generation_falls_back(self, conversations, open_thread):
        writer = FakeWriter(
            bodies=["Would you like me to set Dental Cleaning @ ₹299 up for you first?"]
        )
        service = ReplyService(conversations, SuppressionLedger(), writer)
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "yes do it")

        assert outcome.body_source == "deterministic"
        assert not re_qualifies(outcome.body)

    def test_a_writer_that_raises_does_not_break_the_reply(self, conversations, open_thread):
        writer = FakeWriter(raises=TimeoutError("provider timed out"))
        service = ReplyService(conversations, SuppressionLedger(), writer)
        conversation_id = open_thread()

        outcome = service.handle(conversation_id, "yes do it")

        assert outcome.action == "send"
        assert outcome.body_source == "deterministic"

    def test_the_writer_only_sees_the_conversations_own_facts(self, conversations, open_thread):
        writer = FakeWriter()
        service = ReplyService(conversations, SuppressionLedger(), writer)
        conversation_id = open_thread()

        service.handle(conversation_id, "how much?")

        brief = writer.seen[-1]
        assert brief.purpose == "reply"
        assert brief.intent == QUESTION
        assert brief.facts == make_brief().facts
        assert "suppression" not in str(brief).lower()

    def test_wait_and_end_never_call_the_writer(self, conversations, open_thread):
        writer = FakeWriter()
        service = ReplyService(conversations, SuppressionLedger(), writer)

        service.handle(open_thread("conv_a"), "not now")
        service.handle(open_thread("conv_b"), "not interested")

        assert writer.calls == 0, "no generation for a decision that sends nothing"


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #
class TestOverHttp:
    @pytest.mark.parametrize(
        ("message", "expected_action"),
        [
            ("Yes, do it.", "send"),
            ("not interested", "end"),
            ("not now", "wait"),
            ("how much?", "send"),
            ("Stop messaging me.", "end"),
            (AUTO_REPLY_TEXT, "send"),
        ],
    )
    def test_each_intent_answers_in_contract_shape(self, client, message, expected_action):
        response = client.post(
            "/v1/reply",
            json={"conversation_id": f"conv_{abs(hash(message))}", "merchant_id": "m_001",
                  "from_role": "merchant", "message": message,
                  "received_at": NOW, "turn_number": 2},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["action"] == expected_action
        assert body["rationale"]
        if expected_action == "send":
            assert body["body"].strip()
            assert body["cta"]
        if expected_action == "wait":
            assert body["wait_seconds"] > 0

    def test_a_reply_never_returns_an_empty_send(self, client):
        for message in ["", "  ", "?", "hmm", "yes", "no", AUTO_REPLY_TEXT]:
            response = client.post(
                "/v1/reply",
                json={"conversation_id": "conv_empty_check", "merchant_id": "m_001",
                      "from_role": "merchant", "message": message,
                      "received_at": NOW, "turn_number": 2},
            )
            body = response.json()
            assert response.status_code == 200
            if body["action"] == "send":
                assert body["body"].strip()

    def test_teardown_clears_conversations(self, client):
        client.post(
            "/v1/reply",
            json={"conversation_id": "conv_teardown", "merchant_id": "m_001",
                  "from_role": "merchant", "message": "how much?",
                  "received_at": NOW, "turn_number": 2},
        )
        from state import get_conversation_store

        assert len(get_conversation_store()) == 1
        client.post("/v1/teardown")
        assert len(get_conversation_store()) == 0
