"""Reply intent classification. Deterministic, so every case is a fixed expectation."""

from __future__ import annotations

import pytest

from engine.intent import (
    ACCEPT,
    ACTION_REQUEST,
    AMBIGUOUS,
    AUTO_REPLY,
    DELAY,
    HOSTILE,
    OFF_TOPIC,
    QUESTION,
    REJECT,
    classify_reply,
    looks_like_auto_reply,
    normalise,
)


def intent_of(message: str, **kwargs) -> str:
    return classify_reply(message, **kwargs).intent


class TestAccept:
    @pytest.mark.parametrize(
        "message",
        [
            "yes", "Yes", "YES", "yes.", "yeah", "yep", "sure", "ok", "okay",
            "Ok, let's do it.", "lets do it", "go ahead",
            "Sounds good", "that works", "agreed", "haan", "theek hai",
            "chalega", "kar dijiye", "Yes, do it.", "yes please proceed",
            "Great, go ahead",
        ],
    )
    def test_affirmations(self, message):
        assert intent_of(message) == ACCEPT

    def test_a_commitment_that_asks_what_is_next_is_still_an_acceptance(self):
        """The judge's intent-transition scenario, verbatim."""
        for message in (
            "Ok lets do it. Whats next?",
            "Ok, let's do it. What's next?",
            "yes do it, then what?",
            "sure, how do we start?",
        ):
            assert intent_of(message) == ACCEPT, message

    def test_a_yes_inside_a_refusal_does_not_win(self):
        assert intent_of("yes but not now") == DELAY
        assert intent_of("yes, maybe tomorrow") == DELAY

    def test_a_bare_slot_number_counts_only_when_a_slot_was_offered(self):
        assert intent_of("1") == AMBIGUOUS
        assert intent_of("1", expects_slot=True) == ACCEPT
        assert intent_of("2", expects_slot=True) == ACCEPT


class TestActionRequest:
    @pytest.mark.parametrize(
        "message",
        [
            "activate it", "send it now", "publish it", "do it now",
            "please do", "go ahead and send", "send me the abstract", "share the list",
            "confirm", "proceed", "kar do", "bhej do",
            "Yes please send the abstract. Also draft the patient WhatsApp.",
        ],
    )
    def test_imperatives(self, message):
        assert intent_of(message) in {ACTION_REQUEST, ACCEPT}

    def test_an_imperative_phrased_as_a_question_is_a_question(self):
        assert intent_of("activate it?") == QUESTION


class TestReject:
    @pytest.mark.parametrize(
        "message",
        [
            "no", "No.", "no thanks", "No, thank you", "not interested",
            "skip it", "cancel this", "no need", "don't bother",
            "nahi chahiye", "rehne do", "not required",
        ],
    )
    def test_refusals(self, message):
        assert intent_of(message) == REJECT


class TestDelay:
    @pytest.mark.parametrize(
        "message",
        [
            "not now", "later", "maybe tomorrow", "next week", "in a few days",
            "I'll decide later", "let me think", "give me some time",
            "remind me", "busy right now", "baad mein", "abhi nahi",
            "I'll get back to you",
        ],
    )
    def test_postponements(self, message):
        assert intent_of(message) == DELAY

    def test_a_postponement_is_not_a_refusal(self):
        assert intent_of("not now") != REJECT
        assert intent_of("tomorrow") != REJECT


class TestQuestion:
    @pytest.mark.parametrize(
        "message",
        [
            "how much?", "How much does it cost?", "how does this work?",
            "what do you mean?", "can you explain?", "what is the price",
            "why now?", "kitna?", "kaise hoga?", "what exactly changes?",
        ],
    )
    def test_questions(self, message):
        assert intent_of(message) == QUESTION


class TestAutoReply:
    @pytest.mark.parametrize(
        "message",
        [
            "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
            "Thanks for reaching out. We will get back to you soon.",
            "We have received your message and will revert shortly.",
            "This is an automated reply.",
            "Our team will respond within 24 hours.",
            "I am an automated assistant and cannot help with that.",
            "Main ek automated assistant hoon.",
            "Aapki jaankari ke liye bahut-bahut dhanyavaad.",
            "We are currently closed. Our business hours are 9am to 9pm.",
        ],
    )
    def test_canned_responses(self, message):
        assert intent_of(message) == AUTO_REPLY
        assert looks_like_auto_reply(message) is True

    def test_a_verbatim_repeat_is_machine_behaviour(self):
        text = "Kindly note our team is reviewing this."
        result = classify_reply(text, previous_messages=(text,))

        assert result.intent == AUTO_REPLY
        assert "identical" in result.note

    def test_a_near_repeat_with_different_wording_is_not_forced(self):
        assert classify_reply("ok", previous_messages=("yes",)).intent == ACCEPT

    def test_a_real_reply_is_not_an_auto_reply(self):
        for message in ("yes do it", "how much?", "not interested"):
            assert looks_like_auto_reply(message) is False


class TestHostile:
    @pytest.mark.parametrize(
        "message",
        [
            "Stop messaging me.", "Stop messaging me. This is useless spam.",
            "Why are you bothering me. This is useless. Stop sending these.",
            "don't contact me again", "unsubscribe", "remove my number",
            "leave me alone", "this is rubbish", "bakwas hai",
        ],
    )
    def test_abuse_and_opt_outs(self, message):
        assert intent_of(message) == HOSTILE


class TestOffTopic:
    @pytest.mark.parametrize(
        "message",
        [
            "Btw can you also help me with my GST filing this month?",
            "can you help with my income tax return?",
            "do you give business loans?",
            "I need help with staff salary",
            "can you sort out my electricity bill?",
        ],
    )
    def test_out_of_scope_subjects(self, message):
        assert intent_of(message) == OFF_TOPIC

    def test_subject_beats_shape(self):
        """A question about GST still needs redirecting, not answering."""
        assert intent_of("can you help with GST?") == OFF_TOPIC


class TestAmbiguous:
    @pytest.mark.parametrize("message", ["hmm", "...", "k?", "ðŸ‘", "asdf", ""])
    def test_unreadable_replies(self, message):
        assert intent_of(message) == AMBIGUOUS

    def test_an_empty_message_has_no_confidence(self):
        assert classify_reply("").confidence == 0.0


class TestDeterminism:
    def test_the_same_message_always_classifies_the_same_way(self):
        messages = [
            "yes", "no", "not now", "how much?", "activate it", "hmm",
            "Thank you for contacting us, our team will respond shortly.",
            "Stop messaging me.", "can you help with GST?",
        ]
        first = [classify_reply(m) for m in messages]
        second = [classify_reply(m) for m in messages]

        assert [r.intent for r in first] == [r.intent for r in second]
        assert [r.confidence for r in first] == [r.confidence for r in second]

    def test_whitespace_and_case_do_not_change_the_answer(self):
        assert intent_of("  YES  ") == intent_of("yes") == ACCEPT
        assert normalise("  a   b ") == "a b"
