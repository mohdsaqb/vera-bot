"""Grounding checks on generated text.

Containment, not comprehension: a figure is acceptable only when that exact
figure appears in the brief. Blunt by design — it cannot be argued around.
"""

from __future__ import annotations

import pytest

from engine.types import GenerationBrief
from services.message_validator import (
    MAX_BODY_CHARS,
    MIN_BODY_CHARS,
    validate_generated_body,
)


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
        "facts": (
            "calls are down 50% week-on-week (18 in the last 30 days)",
            "your click-through rate is 2.1% against the 3% dentists median",
        ),
        "offer_title": "Dental Cleaning @ ₹299",
        "offer_is_live": True,
        "ask": "Want me to put Dental Cleaning @ ₹299 live?",
        "cta": "binary_yes_no",
        "taboo_terms": ("guaranteed", "best in city"),
        "deterministic_body": (
            "Dr. Meera — calls are down 50% week-on-week. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        ),
    }
    defaults.update(overrides)
    return GenerationBrief(**defaults)


# --------------------------------------------------------------------------- #
# A-C. shape
# --------------------------------------------------------------------------- #
class TestShape:
    def test_a_grounded_body_passes(self):
        body = (
            "Dr. Meera — calls are down 50% week-on-week, and your click-through "
            "rate is 2.1% against the 3% median. Want me to put Dental Cleaning "
            "@ ₹299 live?"
        )
        assert validate_generated_body(body, make_brief()) == ()

    @pytest.mark.parametrize("body", ["", "   ", "\n"])
    def test_empty_bodies_are_rejected(self, body):
        assert validate_generated_body(body, make_brief()) == ("empty_body",)

    def test_a_truncated_body_is_rejected(self):
        assert "body_too_short" in validate_generated_body("Calls down 50%.", make_brief())

    def test_an_overlong_body_is_rejected(self):
        body = "Dr. Meera — calls are down 50%. " * 40
        problems = validate_generated_body(body, make_brief())

        assert "body_too_long" in problems
        assert len(body) > MAX_BODY_CHARS

    def test_the_length_floor_is_a_real_message_length(self):
        assert MIN_BODY_CHARS < 60 < MAX_BODY_CHARS


# --------------------------------------------------------------------------- #
# D-I. grounding
# --------------------------------------------------------------------------- #
class TestGrounding:
    def test_an_invented_figure_is_caught(self):
        body = (
            "Dr. Meera — calls are down 62% this week. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        problems = validate_generated_body(body, make_brief())

        assert "unsupported_number:62" in problems
        assert "unsupported_percentage:62" in problems

    def test_an_invented_price_is_caught(self):
        body = "Dr. Meera — calls are down 50%. Want me to set up a ₹499 whitening offer?"
        assert "unsupported_amount:499" in validate_generated_body(body, make_brief())

    def test_a_supplied_price_passes(self):
        body = (
            "Dr. Meera — calls are down 50% week-on-week. Want me to put your "
            "Dental Cleaning @ ₹299 live again?"
        )
        assert validate_generated_body(body, make_brief()) == ()

    def test_thousands_separators_compare_equal(self):
        brief = make_brief(facts=("you have 2,410 profile views in 30 days",))
        body = (
            "Dr. Meera — you are at 2410 profile views over 30 days. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        assert validate_generated_body(body, brief) == ()

    def test_an_invented_date_is_caught(self):
        body = (
            "Dr. Meera — calls are down 50%. Want me to put Dental Cleaning @ ₹299 "
            "live before 15 Aug?"
        )
        assert "unsupported_date:15 aug" in validate_generated_body(body, make_brief())

    def test_a_supplied_date_passes(self):
        brief = make_brief(facts=("the new dose limit takes effect 2026-12-15",))
        body = (
            "Dr. Meera — the new dose limit takes effect 2026-12-15. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        assert validate_generated_body(body, brief) == ()

    def test_an_invented_offer_name_is_caught(self):
        body = (
            "Dr. Meera — calls are down 50%. Want me to launch the Summer Smile "
            "Package for you?"
        )
        problems = validate_generated_body(body, make_brief())

        assert any(p.startswith("unsupported_name:Summer") for p in problems)

    def test_an_invented_competitor_is_caught(self):
        body = (
            "Dr. Meera — calls are down 50% because Pearl Dental opened nearby. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        problems = validate_generated_body(body, make_brief())

        assert any(p.startswith("unsupported_name:Pearl") for p in problems)

    def test_a_supplied_name_passes(self):
        brief = make_brief(facts=("Smile Studio opened 1.3 km away on 8 Apr",))
        body = (
            "Dr. Meera — Smile Studio opened 1.3 km away on 8 Apr. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        assert validate_generated_body(body, brief) == ()

    def test_sentence_initial_capitals_are_not_treated_as_names(self):
        body = (
            "Calls are down 50% week-on-week. Worth a look. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        problems = validate_generated_body(body, make_brief())

        assert not [p for p in problems if p.startswith("unsupported_name")]

    def test_a_possessive_form_of_a_cited_name_is_allowed(self):
        """"JIDA's October issue" is the JIDA the brief cited, not a new name."""
        brief = make_brief(
            facts=("a 2,100-patient trial cut caries recurrence 38%",),
            citation="JIDA Oct 2026, p.14",
            offer_title="",
            ask="Want me to draft a patient note?",
        )
        body = (
            "Dr. Meera — JIDA's October issue has a 2,100-patient trial that cut "
            "caries recurrence 38%. Want me to draft a patient note?"
        )
        assert validate_generated_body(body, brief) == ()

    def test_a_genuinely_new_name_is_still_caught(self):
        brief = make_brief(citation="JIDA Oct 2026, p.14")
        body = (
            "Dr. Meera — calls are down 50%, per Lancet's review. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        problems = validate_generated_body(body, brief)

        assert any(p.startswith("unsupported_name:Lancet") for p in problems)

    def test_swapping_the_offer_for_another_is_caught(self):
        """Both words of the substitute can be ordinary English, so no other check
        would notice the decision being overridden."""
        brief = make_brief(
            offer_title="Aligner Consultation @ ₹499",
            offer_is_live=False,
            ask="Want me to put Aligner Consultation @ ₹499 live?",
            deterministic_body=(
                "Dr. Meera — calls are down 50% week-on-week. "
                "Want me to put Aligner Consultation @ ₹499 live?"
            ),
        )
        body = (
            "Dr. Meera — calls are down 50% week-on-week. "
            "Want me to put Free Consultation live?"
        )
        assert "offer_changed_or_dropped" in validate_generated_body(body, brief)

    def test_a_reply_need_not_restate_an_offer_the_draft_did_not_name(self):
        """Mid-thread replies are not obliged to repeat the offer."""
        brief = make_brief(
            purpose="reply",
            ask="Jo slot theek lage reply kar dijiye.",
            deterministic_body=(
                "To be precise: your cleaning is due around 12 Nov. "
                "Jo slot theek lage reply kar dijiye."
            ),
        )
        assert validate_generated_body(brief.deterministic_body, brief) == ()

    def test_dropping_the_offer_entirely_is_caught(self):
        brief = make_brief(offer_title="Dental Cleaning @ ₹299")
        body = "Dr. Meera — calls are down 50% week-on-week. Want me to put something live?"

        assert "offer_changed_or_dropped" in validate_generated_body(body, brief)

    def test_a_suggested_offer_may_not_be_called_live(self):
        """A catalog pattern is a proposal; calling it live is checkable and false."""
        brief = make_brief(offer_is_live=False)
        body = (
            "Dr. Meera — calls are down 50%, and your Dental Cleaning @ ₹299 is "
            "already running. Want me to push it?"
        )
        problems = validate_generated_body(body, brief)

        assert any(p.startswith("suggested_offer_described_as_live") for p in problems)

    def test_a_live_offer_may_be_called_live(self):
        brief = make_brief(offer_is_live=True)
        body = (
            "Dr. Meera — calls are down 50%. Your Dental Cleaning @ ₹299 is already "
            "running — want me to push it in a post?"
        )
        assert validate_generated_body(body, brief) == ()


# --------------------------------------------------------------------------- #
# J. one ask
# --------------------------------------------------------------------------- #
class TestSingleAsk:
    def test_two_questions_are_rejected(self):
        body = (
            "Dr. Meera — calls are down 50%. Want me to put Dental Cleaning @ ₹299 live? "
            "Should I post about it too?"
        )
        assert "multiple_questions" in validate_generated_body(body, make_brief())

    def test_two_requests_are_rejected(self):
        body = (
            "Dr. Meera — calls are down 50%. Shall I put Dental Cleaning @ ₹299 live. "
            "Want me to look at the listing."
        )
        assert "multiple_requests" in validate_generated_body(body, make_brief())

    def test_two_actions_in_one_question_are_rejected(self):
        body = (
            "Dr. Meera — calls are down 50%. Should I activate the offer and update "
            "your listing?"
        )
        assert "multiple_actions_in_one_ask" in validate_generated_body(body, make_brief())

    def test_a_quoted_question_does_not_count(self):
        brief = make_brief(
            facts=(
                "you asked about a corporate thali package — your words: "
                '"what would it look like?"',
            ),
        )
        body = (
            'Dr. Meera — you asked "what would it look like?" about the thali package. '
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        problems = validate_generated_body(body, brief)

        assert "multiple_questions" not in problems

    def test_an_information_only_message_may_not_ask(self):
        brief = make_brief(cta="none", ask="")
        body = "Dr. Meera — calls are down 50% week-on-week. Worth knowing before the weekend?"

        assert "question_added_to_information_only_message" in validate_generated_body(body, brief)


# --------------------------------------------------------------------------- #
# K / L. contradiction and leakage
# --------------------------------------------------------------------------- #
class TestLeakage:
    @pytest.mark.parametrize(
        "body",
        [
            "Dr. Meera — the perf_dip signal fired. Want me to put Dental Cleaning @ ₹299 live?",
            "Dr. Meera — per your suppression key, calls are down 50%. Want me to help out?",
            "Dr. Meera — calls are down 50%. My prompt says to ask: put the offer live?",
            "Dr. Meera — as an AI model I see calls down 50%. Want me to put the offer live?",
        ],
    )
    def test_internal_vocabulary_is_rejected(self, body):
        problems = validate_generated_body(body, make_brief())

        assert any(
            p.startswith(("internal_identifier", "internal_detail")) for p in problems
        ), problems

    def test_taboo_vocabulary_is_rejected(self):
        body = (
            "Dr. Meera — calls are down 50%, and this is guaranteed to fix it. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        assert "taboo_term:guaranteed" in validate_generated_body(body, make_brief())

    def test_a_customer_is_never_shown_merchant_analytics(self):
        brief = make_brief(
            audience="customer",
            recipient_name="Priya",
            facts=("your 6 month cleaning is due around 12 Nov",),
            ask="Reply YES and we will hold a slot for you.",
            deterministic_body="Hi Priya — your cleaning is due around 12 Nov. Reply YES.",
        )
        body = (
            "Hi Priya — your 6 month cleaning is due around 12 Nov, and our "
            "click-through rate is strong. Reply YES and we will hold a slot."
        )
        problems = validate_generated_body(body, brief)

        assert any(p.startswith("merchant_analytics_shown_to_customer") for p in problems)

    def test_a_url_is_rejected(self):
        body = (
            "Dr. Meera — calls are down 50%. Read more at https://example.com. "
            "Want me to put Dental Cleaning @ ₹299 live?"
        )
        assert "contains_url" in validate_generated_body(body, make_brief())
