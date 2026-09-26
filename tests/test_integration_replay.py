"""Replay scenarios and audience separation, driven through the HTTP API.

These mirror the judge's Phase 4 replays (`examples/api-call-examples.md` §4) and
the reply families the phase brief enumerates. Each starts from a real `/v1/tick`
action so the thread has the same state the judge's would.
"""

from __future__ import annotations

import pytest

from conftest import load_contexts, push_context, reply_to, tick_now

DIP = "trg_004_perf_dip_bharat"
BHARAT = "m_002_bharat_dentist_mumbai"
RECALL = "trg_003_recall_due_priya"
MEERA = "m_001_drmeera_dentist_delhi"

AUTO_REPLIES = (
    "Thanks for contacting us.",
    "We have received your message.",
    "We will get back to you shortly.",
    "Thanks for reaching out.",
    "Our team will respond soon.",
)


@pytest.fixture
def thread(client, dataset):
    """A live merchant thread, opened by a real tick action."""
    load_contexts(client, dataset, [DIP])
    actions = tick_now(client, [DIP])
    assert len(actions) == 1
    return actions[0]


# ---------------------------------------------------------------------------- #
# 10. Accept
# ---------------------------------------------------------------------------- #
class TestAccept:
    @pytest.mark.parametrize(
        "message",
        ["Yes.", "Yes, do it", "Go ahead", "Okay, activate it", "Let's do it",
         "ok please proceed", "sure, go ahead"],
    )
    def test_every_phrasing_moves_to_the_action(self, client, thread, message):
        answer = reply_to(client, thread["conversation_id"], message, merchant_id=BHARAT)

        assert answer["action"] == "send"
        assert answer["cta"] == "binary_confirm_cancel"
        assert answer["body"].strip()

    def test_the_accepted_action_keeps_its_context(self, client, thread):
        answer = reply_to(client, thread["conversation_id"], "Yes, do it", merchant_id=BHARAT)

        # The offer proposed in the opening message is the one carried forward.
        assert "Aligner Consultation @ ₹499" in thread["body"]
        assert "Aligner Consultation @ ₹499" in answer["body"]

    def test_the_question_is_not_asked_again(self, client, thread):
        answer = reply_to(client, thread["conversation_id"], "Yes", merchant_id=BHARAT)

        for qualifier in ("would you", "do you", "can you tell", "what if", "how about"):
            assert qualifier not in answer["body"].lower()
        assert answer["body"] != thread["body"]

    def test_no_external_action_is_claimed_as_done(self, client, thread):
        body = reply_to(client, thread["conversation_id"], "Yes, do it",
                     merchant_id=BHARAT)["body"].lower()

        for false_claim in ("i have activated", "i've activated", "is now live",
                            "has been published", "done —", "already live"):
            assert false_claim not in body
        assert "confirm" in body, "the next step is offered, not performed"

    def test_a_second_acceptance_does_not_repeat_the_same_body(self, client, thread):
        first = reply_to(client, thread["conversation_id"], "yes", merchant_id=BHARAT)
        second = reply_to(client, thread["conversation_id"], "yes", merchant_id=BHARAT)

        if second["action"] == "send":
            assert second["body"] != first["body"]


# ---------------------------------------------------------------------------- #
# 11. Reject
# ---------------------------------------------------------------------------- #
class TestReject:
    @pytest.mark.parametrize(
        "message", ["No", "Don't do this", "Not interested", "Skip it", "no thanks"]
    )
    def test_a_refusal_closes_without_pushing(self, client, thread, message):
        answer = reply_to(client, thread["conversation_id"], message, merchant_id=BHARAT)

        assert answer["action"] == "end"
        assert "body" not in answer or not answer.get("body")

    def test_the_declined_action_is_not_offered_again(self, client, dataset, thread):
        reply_to(client, thread["conversation_id"], "Not interested", merchant_id=BHARAT)

        push_context(client, "trigger", DIP, dataset.triggers[DIP])
        assert tick_now(client, [DIP]) == [], "the declined story must not return"

    def test_a_refusal_silences_the_story_not_the_merchant(self, client, dataset, thread):
        """§2.6 suppresses the conversation; §4.3 suppresses the merchant.

        Conflating them stops every other story, including customer-facing ones,         over one declined offer.
        """
        reply_to(client, thread["conversation_id"], "Not interested", merchant_id=BHARAT)

        # The declined story stays closed...
        push_context(client, "trigger", DIP, dataset.triggers[DIP])
        assert tick_now(client, [DIP]) == []

        # ...but a different story for the same merchant is still reachable.
        load_contexts(client, dataset, ["trg_005_renewal_due_bharat"])
        from state import get_suppression_ledger

        assert not get_suppression_ledger().is_suppressed(f"optout:{BHARAT}")

    def test_hostility_does_silence_the_merchant(self, client, dataset, thread):
        reply_to(client, thread["conversation_id"], "Stop messaging me.",
                 merchant_id=BHARAT)

        from state import get_suppression_ledger

        assert get_suppression_ledger().is_suppressed(f"optout:{BHARAT}")
        load_contexts(client, dataset, ["trg_005_renewal_due_bharat"])
        assert tick_now(client, ["trg_005_renewal_due_bharat"]) == []

    def test_a_later_message_on_a_closed_thread_stays_closed(self, client, thread):
        reply_to(client, thread["conversation_id"], "Not interested", merchant_id=BHARAT)

        answer = reply_to(client, thread["conversation_id"], "actually yes", merchant_id=BHARAT)

        assert answer["action"] == "end"


# ---------------------------------------------------------------------------- #
# 12. Not now
# ---------------------------------------------------------------------------- #
class TestNotNow:
    @pytest.mark.parametrize(
        "message", ["Not now", "Later", "Maybe tomorrow", "I'll do it later",
                    "give me some time", "busy right now"]
    )
    def test_a_postponement_waits(self, client, thread, message):
        answer = reply_to(client, thread["conversation_id"], message, merchant_id=BHARAT)

        assert answer["action"] == "wait"
        assert answer["wait_seconds"] > 0
        assert "body" not in answer

    def test_it_is_not_treated_as_a_refusal(self, client, thread):
        from state import get_conversation_store

        reply_to(client, thread["conversation_id"], "not now", merchant_id=BHARAT)

        state = get_conversation_store().get(thread["conversation_id"])
        assert state.active is True
        assert state.should_end is False
        assert state.session_state == "holding"

    def test_the_proposal_is_not_immediately_repeated(self, client, thread):
        reply_to(client, thread["conversation_id"], "not now", merchant_id=BHARAT)

        # Nothing is sent in the same turn, and the story stays suppressed.
        assert tick_now(client, [DIP]) == []

    def test_tomorrow_waits_longer_than_later(self, client, dataset):
        load_contexts(client, dataset, [DIP, "trg_021_unverified_gbp_sunrise"])
        soon = reply_to(client, tick_now(client, [DIP])[0]["conversation_id"], "later",
                     merchant_id=BHARAT)
        later = reply_to(client, tick_now(client, ["trg_021_unverified_gbp_sunrise"])[0][
            "conversation_id"], "maybe tomorrow", merchant_id="m_010")

        assert later["wait_seconds"] > soon["wait_seconds"]


# ---------------------------------------------------------------------------- #
# 13. Questions
# ---------------------------------------------------------------------------- #
class TestQuestion:
    @pytest.mark.parametrize(
        "message",
        ["How much is it?", "What does this mean?", "Why are bookings down?",
         "Which offer?", "can you explain?"],
    )
    def test_a_question_is_answered_from_context(self, client, thread, message):
        answer = reply_to(client, thread["conversation_id"], message, merchant_id=BHARAT)

        assert answer["action"] == "send"
        assert answer["body"].strip()

    def test_the_answer_uses_the_supplied_facts(self, client, thread):
        answer = reply_to(client, thread["conversation_id"], "Why are bookings down?",
                       merchant_id=BHARAT)

        # The fact the opening message was built on, restated.
        assert "50%" in answer["body"]

    def test_nothing_is_invented_in_an_answer(self, client, thread):
        answer = reply_to(client, thread["conversation_id"], "How much is it?",
                       merchant_id=BHARAT)

        # Only figures from the thread may appear.
        import re
        allowed = thread["body"] + " " + " ".join(thread["template_params"])
        for number in re.findall(r"\d[\d,.]*", answer["body"]):
            assert number.rstrip(".") in allowed.replace(",", "") or number in allowed

    def test_an_unanswerable_question_says_so(self, client):
        answer = reply_to(client, "conv_no_context_at_all", "How much is it?",
                       merchant_id="m_404")

        assert answer["action"] == "send"
        assert "don't have that" in answer["body"].lower()


# ---------------------------------------------------------------------------- #
# 14. Auto-reply loops
# ---------------------------------------------------------------------------- #
class TestAutoReplyLoop:
    @pytest.mark.parametrize("canned", AUTO_REPLIES)
    def test_each_canned_form_is_recognised(self, client, dataset, canned):
        load_contexts(client, dataset, [DIP])
        conversation_id = tick_now(client, [DIP])[0]["conversation_id"]

        answer = reply_to(client, conversation_id, canned, merchant_id=BHARAT)

        assert answer["action"] == "send"
        assert "auto-responder" in answer["body"]

    def test_a_repeated_canned_reply_stops_the_loop(self, client, thread):
        canned = AUTO_REPLIES[0]
        actions = [
            reply_to(client, thread["conversation_id"], canned, merchant_id=BHARAT)["action"]
            for _ in range(5)
        ]

        assert actions[0] == "send"
        assert actions.count("send") == 1, "at most one turn spent on a machine"
        assert actions[-1] == "end"

    def test_varied_canned_replies_also_terminate(self, client, thread):
        actions = [
            reply_to(client, thread["conversation_id"], canned, merchant_id=BHARAT)["action"]
            for canned in AUTO_REPLIES
        ]

        assert "end" in actions
        assert actions.count("send") <= 1

    def test_canned_text_is_not_read_as_engagement(self, client, thread):
        from state import get_conversation_store

        reply_to(client, thread["conversation_id"], AUTO_REPLIES[0], merchant_id=BHARAT)

        state = get_conversation_store().get(thread["conversation_id"])
        assert state.has_real_engagement is False
        assert state.consecutive_auto_replies == 1

    def test_the_owner_can_still_break_through(self, client, thread):
        reply_to(client, thread["conversation_id"], AUTO_REPLIES[0], merchant_id=BHARAT)

        answer = reply_to(client, thread["conversation_id"], "yes go ahead", merchant_id=BHARAT)

        assert answer["action"] == "send"
        assert "CONFIRM" in answer["body"]


# ---------------------------------------------------------------------------- #
# 15. Off topic
# ---------------------------------------------------------------------------- #
class TestOffTopic:
    @pytest.mark.parametrize(
        "message",
        ["Btw can you also help me with my GST filing this month?",
         "do you give business loans?", "can you sort out my electricity bill?",
         "I need help with staff salary"],
    )
    def test_a_detour_is_declined_not_attempted(self, client, thread, message):
        answer = reply_to(client, thread["conversation_id"], message, merchant_id=BHARAT)

        assert answer["action"] == "send"
        assert "outside what I can help with" in answer["body"]

    def test_the_active_action_is_not_executed_by_a_detour(self, client, thread):
        answer = reply_to(client, thread["conversation_id"], "can you file my GST?",
                       merchant_id=BHARAT)

        assert "CONFIRM" not in answer["body"], "a detour must not trigger the action"

    def test_the_thread_survives_a_detour(self, client, thread):
        from state import get_conversation_store

        reply_to(client, thread["conversation_id"], "can you help with GST?", merchant_id=BHARAT)
        answer = reply_to(client, thread["conversation_id"], "ok do it", merchant_id=BHARAT)

        assert get_conversation_store().get(thread["conversation_id"]).active is True
        assert answer["action"] == "send"


# ---------------------------------------------------------------------------- #
# 16. Ambiguous
# ---------------------------------------------------------------------------- #
class TestAmbiguous:
    @pytest.mark.parametrize("message", ["maybe", "hmm", "okay?", "what?", "..."])
    def test_ambiguity_is_never_read_as_acceptance(self, client, dataset, message):
        load_contexts(client, dataset, [DIP])
        conversation_id = tick_now(client, [DIP])[0]["conversation_id"]

        answer = reply_to(client, conversation_id, message, merchant_id=BHARAT)

        assert answer["action"] in {"send", "wait"}
        if answer["action"] == "send":
            assert "CONFIRM" not in answer["body"], f"{message!r} executed the action"

    def test_fine_is_read_as_agreement_but_okay_question_is_not(self, client, dataset):
        """"fine" commits; "okay?" is asking, and the two must part company."""
        load_contexts(client, dataset, [DIP, "trg_021_unverified_gbp_sunrise"])
        agreed = reply_to(client, tick_now(client, [DIP])[0]["conversation_id"], "fine",
                       merchant_id=BHARAT)
        asked = reply_to(client, tick_now(client, ["trg_021_unverified_gbp_sunrise"])[0][
            "conversation_id"], "okay?", merchant_id="m_010")

        assert "CONFIRM" in agreed["body"]
        assert "CONFIRM" not in asked.get("body", "")

    def test_repeated_ambiguity_backs_off_then_closes(self, client, thread):
        actions = [
            reply_to(client, thread["conversation_id"], "hmm", merchant_id=BHARAT)["action"]
            for _ in range(4)
        ]

        assert actions[0] == "send"
        assert "wait" in actions or "end" in actions
        assert actions[-1] in {"wait", "end"}


# ---------------------------------------------------------------------------- #
# 17. Customer versus merchant separation
# ---------------------------------------------------------------------------- #
class TestAudienceSeparation:
    MERCHANT_ONLY = (
        "click-through", "percentage points", "peer", "median", "profile views",
        "ctr", "benchmark", "suppression", "trigger", "rationale", "priority",
    )

    def test_a_customer_message_carries_no_merchant_analytics(self, client, dataset):
        load_contexts(client, dataset, [RECALL])
        actions = tick_now(client, [RECALL])

        assert len(actions) == 1
        action = actions[0]
        assert action["send_as"] == "merchant_on_behalf"
        lowered = action["body"].lower()
        for term in self.MERCHANT_ONLY:
            assert term not in lowered, f"leaked {term!r} to a customer"

    def test_a_customer_message_uses_customer_facts(self, client, dataset):
        load_contexts(client, dataset, [RECALL])
        body = tick_now(client, [RECALL])[0]["body"]

        assert "Priya" in body
        assert "Wed 5 Nov, 6pm" in body, "the real slots from the trigger payload"
        assert "Dental Cleaning @ ₹299" in body, "the merchant's own live offer"

    def test_no_internal_rationale_reaches_a_customer(self, client, dataset):
        load_contexts(client, dataset, [RECALL])
        action = tick_now(client, [RECALL])[0]

        assert action["rationale"], "the judge still gets the reasoning"
        assert action["rationale"] not in action["body"]
        assert "urgency" not in action["body"].lower()

    def test_consent_is_respected_over_http(self, client, dataset):
        import copy

        promotional_only = copy.deepcopy(dataset.customers["c_001_priya_for_m001"])
        promotional_only["consent"]["scope"] = ["promotional_offers"]
        load_contexts(client, dataset, [RECALL])
        push_context(client, "customer", "c_001_priya_for_m001", promotional_only, 2)
        push_context(client, "trigger", RECALL, dataset.triggers[RECALL], 2)

        assert tick_now(client, [RECALL]) == [], "a clinical recall needs clinical consent"

    def test_merchant_messages_may_use_merchant_metrics(self, client, dataset):
        load_contexts(client, dataset, [DIP])
        body = tick_now(client, [DIP])[0]["body"]

        assert "click-through rate" in body
        assert "percentage points" in body

    def test_a_customer_reply_stays_customer_facing(self, client, dataset):
        load_contexts(client, dataset, [RECALL])
        action = tick_now(client, [RECALL])[0]

        answer = reply_to(client, action["conversation_id"], "how much?",
                       merchant_id=MEERA, customer_id="c_001_priya_for_m001",
                       from_role="customer")

        assert answer["action"] == "send"
        lowered = answer["body"].lower()
        for term in self.MERCHANT_ONLY:
            assert term not in lowered
