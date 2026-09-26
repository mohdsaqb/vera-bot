"""`compose()` end to end — the A-O scenario matrix from the phase brief.

Each test names the behaviour it pins rather than a specific sentence, so the
wording can improve without the suite needing edits; what is asserted is that
the right fact, offer, CTA, sender and decision come out.
"""

from __future__ import annotations

import json
import re

from conftest import deep_set, variant
from engine import compose, compose_dict, policy
from engine.render import validate_body

CTA_VALUES = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}


def _compose(dataset, trigger_id, now=None, **swaps):
    category, merchant, trigger, customer = dataset.compose_args(trigger_id)
    return compose(
        swaps.get("category", category),
        swaps.get("merchant", merchant),
        swaps.get("trigger", trigger),
        swaps.get("customer", customer),
        now=now,
    )


# --------------------------------------------------------------------------- #
# A. one trigger, one obvious signal
# --------------------------------------------------------------------------- #
class TestObviousSignal:
    def test_research_digest_uses_the_supplied_research(self, dataset, now):
        result = _compose(dataset, "trg_001_research_digest_dentists", now)

        assert result.should_send
        # The specific numbers and the citation, both from the category digest.
        assert "38%" in result.body
        assert "2,100" in result.body
        assert "JIDA Oct 2026, p.14" in result.body
        # Tied to this merchant's own cohort.
        assert "124" in result.body
        assert result.send_as == "vera"
        assert result.cta in CTA_VALUES

    def test_plan_carries_provenance_for_every_fact(self, dataset, now):
        plan = _compose(dataset, "trg_001_research_digest_dentists", now).plan

        assert plan is not None
        assert plan.primary_fact.source.startswith("category.digest")
        assert all(fact.source for fact in plan.facts)
        assert plan.evidence
        assert plan.citations == ("JIDA Oct 2026, p.14",)


# --------------------------------------------------------------------------- #
# B. multiple triggers -> the strongest is chosen deterministically
# --------------------------------------------------------------------------- #
class TestTriggerChoice:
    def test_priority_reflects_urgency_and_support(self, dataset, now):
        urgent = _compose(dataset, "trg_018_supply_atorvastatin_recall", now)
        idle = _compose(dataset, "trg_012_milestone_mylari", now)

        assert urgent.plan.priority > idle.plan.priority

    def test_each_trigger_produces_its_own_story(self, dataset, now):
        digest = _compose(dataset, "trg_001_research_digest_dentists", now)
        competitor = _compose(dataset, "trg_023_competitor_opened_dentist", now)

        # Same merchant, different reason for writing, different message.
        assert digest.merchant_id == competitor.merchant_id
        assert digest.body != competitor.body
        assert digest.suppression_key != competitor.suppression_key
        assert digest.conversation_id != competitor.conversation_id


# --------------------------------------------------------------------------- #
# C / D. performance dip and spike use the real metric
# --------------------------------------------------------------------------- #
class TestPerformance:
    def test_dip_quotes_the_actual_decline(self, dataset, now):
        result = _compose(dataset, "trg_004_perf_dip_bharat", now)

        assert result.should_send
        assert "50%" in result.body
        assert "down" in result.body.lower()
        assert "12" in result.body  # the supplied baseline

    def test_dip_pairs_with_the_peer_gap_in_percentage_points(self, dataset, now):
        result = _compose(dataset, "trg_004_perf_dip_bharat", now)
        assert "percentage points" in result.body

    def test_spike_is_framed_as_an_opportunity_with_its_driver(self, dataset, now):
        result = _compose(dataset, "trg_024_perf_spike_zen", now)

        assert result.should_send
        assert "15%" in result.body
        assert "up" in result.body.lower()
        assert "kids yoga" in result.body

    def test_expected_seasonal_dip_is_reframed_not_alarmed(self, dataset, now):
        result = _compose(dataset, "trg_014_seasonal_acquisition_dip_powerhouse", now)

        assert result.should_send
        assert "seasonal" in result.body.lower()
        assert "30%" in result.body

    def test_a_dip_trigger_on_rising_numbers_sends_nothing(self, dataset, now):
        """The merchant can check this: claiming a fall that did not happen is
        the one thing that must never happen."""
        rising = deep_set(dataset.merchants["m_002_bharat_dentist_mumbai"],
                          "performance.delta_7d.calls_pct", 0.25)
        result = _compose(dataset, "trg_004_perf_dip_bharat", now,
                          merchant=variant(rising, signals=[]),
                          trigger=variant(dataset.triggers["trg_004_perf_dip_bharat"],
                                          payload={"metric": "calls", "window": "7d"}))

        assert not result.should_send
        assert result.reason_code == policy.REASON_CONTRADICTED


# --------------------------------------------------------------------------- #
# E / F. offers
# --------------------------------------------------------------------------- #
class TestOffers:
    def test_an_existing_active_offer_is_used_verbatim(self, dataset, now):
        result = _compose(dataset, "trg_003_recall_due_priya", now)

        assert "Dental Cleaning @ ₹299" in result.body
        assert result.plan.offer.is_existing is True

    def test_no_offer_is_invented_when_none_is_available(self, dataset, now):
        bare_category = variant(dataset.categories["dentists"], offer_catalog=[])
        bare_merchant = variant(dataset.merchants["m_002_bharat_dentist_mumbai"], offers=[])
        result = _compose(dataset, "trg_004_perf_dip_bharat", now,
                          category=bare_category, merchant=bare_merchant)

        assert result.should_send
        assert result.plan.offer is None
        assert "₹" not in result.body
        assert "%" in result.body  # still specific, via the real metric

    def test_no_price_appears_that_was_not_supplied(self, dataset, now):
        """Every rupee figure in a body must come from an input context."""
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if not result.should_send:
                continue
            category, merchant, _, _ = dataset.compose_args(trigger_id)
            supplied = " ".join(
                [o["title"] for o in merchant.get("offers", [])]
                + [o["title"] for o in category.get("offer_catalog", [])]
                + [json.dumps(item, ensure_ascii=False) for item in category.get("digest", [])]
                + [str(v) for v in _numbers_in_payloads(dataset, trigger_id)]
            )
            for amount in re.findall(r"₹[\d,]+", result.body):
                assert amount in supplied, f"{trigger_id}: unsourced amount {amount}"


def _numbers_in_payloads(dataset, trigger_id):
    """Every scalar the trigger payload supplies, for provenance checks."""
    trigger = dataset.triggers[trigger_id]

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        else:
            yield value

    return list(walk(trigger.get("payload", {})))


# --------------------------------------------------------------------------- #
# G / H. knowledge and occasion triggers
# --------------------------------------------------------------------------- #
class TestSuppliedContext:
    def test_compliance_uses_the_real_rule_and_cites_it(self, dataset, now):
        result = _compose(dataset, "trg_002_compliance_dci_radiograph", now)

        assert result.should_send
        assert "1.0 mSv" in result.body
        assert "Dental Council of India circular 2026-11-04" in result.body

    def test_supply_alert_names_the_real_batches(self, dataset, now):
        result = _compose(dataset, "trg_018_supply_atorvastatin_recall", now)

        assert "atorvastatin" in result.body
        assert "AT2024-1102" in result.body and "AT2024-1108" in result.body

    def test_a_local_chapter_item_says_so(self, dataset, now):
        """The IDA Delhi calendar item belongs to Dr. Meera's own city."""
        result = _compose(dataset, "trg_022_cde_webinar_dentists", now)

        assert result.plan.supporting_fact.kind == "local_chapter"
        assert "Delhi" in result.body

    def test_local_relevance_is_not_claimed_for_another_city(self, dataset, now):
        elsewhere = deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"],
                             "identity.city", "Chennai")
        result = _compose(dataset, "trg_022_cde_webinar_dentists", now, merchant=elsewhere)

        assert result.should_send
        assert "Chennai chapter" not in result.body

    def test_compliance_anchors_on_who_the_item_touches(self, dataset, now):
        """A product recall is about the customers who bought it; a rule change
        is about the practice, so each takes a different second fact."""
        recall = _compose(dataset, "trg_018_supply_atorvastatin_recall", now)
        rule = _compose(dataset, "trg_002_compliance_dci_radiograph", now)

        assert recall.plan.supporting_fact.kind == "customer_base"
        assert "240" in recall.body
        assert rule.plan.supporting_fact.kind == "digest_action"
        assert "X-ray setup" in rule.body

    def test_cde_invite_uses_the_date_and_credits(self, dataset, now):
        result = _compose(dataset, "trg_022_cde_webinar_dentists", now)

        assert "2 May" in result.body
        assert "2 CDE credits" in result.body

    def test_match_day_uses_the_real_fixture_and_the_category_pattern(self, dataset, now):
        result = _compose(dataset, "trg_010_ipl_match_delhi", now)

        assert "DC vs MI" in result.body
        assert "Arun Jaitley" in result.body
        assert "7:30pm" in result.body
        # The payload says it is not a weeknight and the category beat says
        # match-night promos belong Tue-Thu, so the advice is the contrarian one.
        assert "Tue-Thu" in result.body

    def test_seasonal_demand_shift_uses_the_supplied_trends(self, dataset, now):
        result = _compose(dataset, "trg_020_summer_demand_shift", now)

        assert "ORS" in result.body
        assert "sunscreen" in result.body

    def test_a_festival_too_far_out_is_not_worth_a_message(self, dataset, now):
        result = _compose(dataset, "trg_006_festival_diwali", now)

        assert not result.should_send
        assert result.reason_code == policy.REASON_EVENT_DISTANT

    def test_a_near_festival_is_acted_on(self, dataset, now):
        near = variant(
            dataset.triggers["trg_006_festival_diwali"],
            payload={"festival": "Diwali", "date": "2026-05-10", "days_until": 14,
                     "category_relevance": ["salons"]},
        )
        result = _compose(dataset, "trg_006_festival_diwali", now, trigger=near)

        assert result.should_send
        assert "Diwali" in result.body
        assert "14 days" in result.body

    def test_an_unnamed_competitor_is_never_invented(self, dataset, now):
        blank = variant(dataset.triggers["trg_023_competitor_opened_dentist"],
                        payload={"placeholder": True})
        result = _compose(dataset, "trg_023_competitor_opened_dentist", now, trigger=blank)

        assert not result.should_send
        assert result.reason_code == policy.REASON_CONTRADICTED


# --------------------------------------------------------------------------- #
# I. customer outreach
# --------------------------------------------------------------------------- #
class TestCustomerOutreach:
    def test_recall_is_sent_from_the_merchants_number(self, dataset, now):
        result = _compose(dataset, "trg_003_recall_due_priya", now)

        assert result.send_as == "merchant_on_behalf"
        assert result.customer_id == "c_001_priya_for_m001"
        assert "Priya" in result.body
        assert "Dr. Meera's Dental Clinic" in result.body

    def test_real_slots_are_offered_with_a_slot_cta(self, dataset, now):
        result = _compose(dataset, "trg_003_recall_due_priya", now)

        assert "Wed 5 Nov, 6pm" in result.body
        assert "Thu 6 Nov, 5pm" in result.body
        assert result.cta == "multi_choice_slot"

    def test_hindi_preference_is_honoured(self, dataset, now):
        """Priya's language_pref is "hi-en mix"; pure English is an anti-pattern."""
        result = _compose(dataset, "trg_003_recall_due_priya", now)
        assert any(word in result.body for word in ("Namaste", "kijiye", "hain", "dijiye"))

    def test_english_preference_stays_english(self, dataset, now):
        result = _compose(dataset, "trg_015_winback_rashmi", now)

        assert result.plan.language == "en"
        assert "Reply YES" in result.body

    def test_refill_confirms_with_the_real_molecules(self, dataset, now):
        result = _compose(dataset, "trg_019_chronic_refill_grandfather", now)

        assert result.cta == "binary_confirm_cancel"
        for molecule in ("metformin", "atorvastatin", "telmisartan"):
            assert molecule in result.body
        assert "CONFIRM" in result.body

    def test_winback_addresses_the_customers_own_goal(self, dataset, now):
        result = _compose(dataset, "trg_015_winback_rashmi", now)

        assert "weight loss" in result.body
        # The exact figure from trigger.payload.days_since_last_visit, not a
        # rounded "about 8 weeks" — the gap is the fact, so it is stated precisely.
        assert "57 days" in result.body
        assert result.cta == "binary_yes_no"

    def test_no_merchant_analytics_leak_to_a_customer(self, dataset, now):
        """A customer must never be shown the shop's internal performance."""
        leaks = ("click-through", "percentage points", "peer", "median", "profile views")
        for trigger_id, trigger in dataset.triggers.items():
            if trigger.get("scope") != "customer":
                continue
            result = _compose(dataset, trigger_id, now)
            if not result.should_send:
                continue
            lowered = result.body.lower()
            for term in leaks:
                assert term not in lowered, f"{trigger_id} leaked '{term}' to a customer"

    def test_missing_consent_for_the_purpose_sends_nothing(self, dataset, now):
        promotional_only = deep_set(dataset.customers["c_001_priya_for_m001"],
                                    "consent.scope", ["promotional_offers"])
        result = _compose(dataset, "trg_003_recall_due_priya", now, customer=promotional_only)

        assert not result.should_send
        assert result.reason_code == policy.REASON_NO_CONSENT

    def test_opted_out_customer_sends_nothing(self, dataset, now):
        opted_out = deep_set(dataset.customers["c_001_priya_for_m001"],
                             "preferences.reminder_opt_in", False)
        result = _compose(dataset, "trg_003_recall_due_priya", now, customer=opted_out)

        assert result.reason_code == policy.REASON_OPTED_OUT

    def test_unreachable_customer_sends_nothing(self, dataset, now):
        # A walk-in with no recorded phone and no channel, like c_015 in the
        # seed data, but attached to this trigger's own merchant.
        unreachable = deep_set(dataset.customers["c_013_grandfather_for_m009"],
                               "identity.phone_redacted", None)
        unreachable = deep_set(unreachable, "preferences.channel", "none_recorded")
        result = _compose(dataset, "trg_019_chronic_refill_grandfather", now,
                          customer=unreachable)

        assert result.reason_code == policy.REASON_NOT_CONTACTABLE

    def test_a_customer_of_another_merchant_is_ignored(self, dataset, now):
        result = _compose(dataset, "trg_001_research_digest_dentists", now,
                          customer=dataset.customers["c_010_rashmi_for_m007"])

        assert result.should_send
        assert result.customer_id is None
        assert "Rashmi" not in result.body


# --------------------------------------------------------------------------- #
# J. no-action
# --------------------------------------------------------------------------- #
class TestNoAction:
    def test_no_action_still_explains_itself(self, dataset):
        result = _compose(dataset, "trg_010_ipl_match_delhi", now="2026-05-05T00:00:00Z")

        assert result.decision == "no_action"
        assert result.body == ""
        assert result.rationale
        assert result.plan is None

    def test_expired_trigger(self, dataset):
        result = _compose(dataset, "trg_010_ipl_match_delhi", now="2026-06-01T00:00:00Z")
        assert result.reason_code == policy.REASON_EXPIRED

    def test_wrong_category_for_the_trigger_kind(self, dataset, now):
        result = _compose(dataset, "trg_019_chronic_refill_grandfather", now,
                          merchant=dataset.merchants["m_001_drmeera_dentist_delhi"])
        assert result.reason_code == policy.REASON_CATEGORY_MISMATCH

    def test_missing_required_context(self, dataset, now):
        assert _compose(dataset, "trg_001_research_digest_dentists", now,
                        category=None).reason_code == policy.REASON_MISSING_CONTEXT
        assert _compose(dataset, "trg_001_research_digest_dentists", now,
                        merchant=None).reason_code == policy.REASON_MISSING_CONTEXT
        assert compose(None, None, None).reason_code == policy.REASON_MISSING_CONTEXT

    def test_placeholder_payload_for_a_payload_dependent_family(self, dataset, now):
        blank = variant(dataset.triggers["trg_012_milestone_mylari"],
                        payload={"placeholder": True, "metric_or_topic": "milestone_reached"})
        result = _compose(dataset, "trg_012_milestone_mylari", now, trigger=blank)

        assert not result.should_send
        assert result.reason_code in {policy.REASON_NO_FACT, policy.REASON_CONTRADICTED}


# --------------------------------------------------------------------------- #
# K. suppression keys
# --------------------------------------------------------------------------- #
class TestSuppression:
    def test_the_triggers_own_key_is_echoed(self, dataset, now):
        result = _compose(dataset, "trg_001_research_digest_dentists", now)
        assert result.suppression_key == "research:dentists:2026-W17"

    def test_the_same_event_always_produces_the_same_key(self, dataset, now):
        first = _compose(dataset, "trg_003_recall_due_priya", now)
        second = _compose(dataset, "trg_003_recall_due_priya", now)

        assert first.suppression_key == second.suppression_key
        assert first.conversation_id == second.conversation_id

    def test_a_key_is_synthesised_when_the_trigger_has_none(self, dataset, now):
        keyless = variant(dataset.triggers["trg_004_perf_dip_bharat"], suppression_key="")
        result = _compose(dataset, "trg_004_perf_dip_bharat", now, trigger=keyless)

        assert result.suppression_key
        assert "m_002_bharat_dentist_mumbai" in result.suppression_key
        assert "performance" in result.suppression_key

    def test_different_events_do_not_share_a_key(self, dataset, now):
        keys = set()
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if result.should_send:
                keys.add(result.suppression_key)
        sends = sum(1 for t in dataset.triggers if _compose(dataset, t, now).should_send)

        assert len(keys) == sends

    def test_conversation_ids_are_readable_and_decodable(self, dataset, now):
        result = _compose(dataset, "trg_003_recall_due_priya", now)

        assert result.conversation_id.startswith("conv_m_001")
        assert "c_001" in result.conversation_id
        assert "recall" in result.conversation_id
        assert len(result.conversation_id) < 80


# --------------------------------------------------------------------------- #
# L. category differences
# --------------------------------------------------------------------------- #
class TestCategoryFit:
    def test_the_five_categories_do_not_share_a_template(self, dataset, now):
        # One curiosity-family message per category: same family, five voices.
        bodies = {}
        for merchant_id, merchant in dataset.merchants.items():
            trigger = variant(dataset.triggers["trg_008_curious_ask_studio11"],
                              merchant_id=merchant_id, suppression_key=f"curious:{merchant_id}")
            result = compose(dataset.category_for(merchant_id), merchant, trigger, None, now=now)
            if result.should_send:
                bodies.setdefault(merchant["category_slug"], []).append(result.body)

        assert len(bodies) >= 4
        flat = [body for group in bodies.values() for body in group]
        assert len(set(flat)) == len(flat)

    def test_dentists_get_the_clinical_honorific(self, dataset, now):
        result = _compose(dataset, "trg_001_research_digest_dentists", now)
        assert result.body.startswith("Dr. Meera")

    def test_an_honorific_is_not_doubled(self, dataset, now):
        """Generated dentist merchants already carry "Dr." in owner_first_name."""
        merchant = deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"],
                            "identity.owner_first_name", "Dr. Asha")
        result = _compose(dataset, "trg_001_research_digest_dentists", now, merchant=merchant)

        assert result.body.startswith("Dr. Asha")
        assert "Dr. Dr." not in result.body

    def test_owner_first_name_is_used_when_present(self, dataset, now):
        for trigger_id in ("trg_004_perf_dip_bharat", "trg_010_ipl_match_delhi",
                           "trg_021_unverified_gbp_sunrise"):
            _, merchant, _, _ = dataset.compose_args(trigger_id)
            result = _compose(dataset, trigger_id, now)
            if result.should_send:
                assert merchant["identity"]["owner_first_name"] in result.body

    def test_no_taboo_vocabulary_anywhere(self, dataset, now):
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if result.should_send:
                assert validate_body(result.body, result.plan) == ()


# --------------------------------------------------------------------------- #
# M / N. optional and sparse context
# --------------------------------------------------------------------------- #
class TestPartialContext:
    def test_customer_argument_is_optional(self, dataset, now):
        category, merchant, trigger, _ = dataset.compose_args("trg_001_research_digest_dentists")
        result = compose(category, merchant, trigger, now=now)

        assert result.should_send
        assert result.customer_id is None

    def test_now_is_optional(self, dataset):
        result = _compose(dataset, "trg_001_research_digest_dentists")

        assert result.should_send
        assert result.plan.priority > 0

    def test_a_merchant_with_only_identity_does_not_crash(self, dataset, now):
        sparse = {"merchant_id": "m_x", "category_slug": "dentists",
                  "identity": {"name": "X Dental", "owner_first_name": "Ravi"}}
        result = _compose(dataset, "trg_001_research_digest_dentists", now, merchant=sparse)

        assert result.should_send
        assert "Ravi" in result.body

    def test_every_seed_trigger_composes_without_raising(self, dataset, now):
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            assert result.decision in {"send", "no_action"}
            assert result.rationale

    def test_empty_payload_blocks_are_tolerated(self, dataset, now):
        stripped = variant(
            dataset.merchants["m_001_drmeera_dentist_delhi"],
            offers=[], signals=[], review_themes=[], conversation_history=[],
            customer_aggregate={},
        )
        result = _compose(dataset, "trg_001_research_digest_dentists", now, merchant=stripped)
        assert result.decision in {"send", "no_action"}


# --------------------------------------------------------------------------- #
# O. determinism
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_repeated_calls_are_identical(self, dataset, now):
        for trigger_id in dataset.triggers:
            first = _compose(dataset, trigger_id, now)
            second = _compose(dataset, trigger_id, now)

            assert first.to_dict() == second.to_dict()
            assert first.conversation_id == second.conversation_id
            assert first.template_params == second.template_params

    def test_dict_form_matches_the_submission_contract(self, dataset, now):
        category, merchant, trigger, customer = dataset.compose_args("trg_003_recall_due_priya")
        result = compose_dict(category, merchant, trigger, customer, now=now)

        assert set(result) == {"body", "cta", "send_as", "suppression_key", "rationale"}
        assert result["send_as"] == "merchant_on_behalf"


# --------------------------------------------------------------------------- #
# Message-shape rules that apply to every send
# --------------------------------------------------------------------------- #
class TestMessageShape:
    def test_single_cta_and_no_urls(self, dataset, now):
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if not result.should_send:
                continue
            assert "http" not in result.body and "www." not in result.body
            assert result.cta in CTA_VALUES
            unquoted = re.sub(r'"[^"]*"', "", result.body)
            assert unquoted.count("?") <= 1, f"{trigger_id} asks more than one question"

    def test_the_ask_is_in_the_closing_sentence(self, dataset, now):
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if not result.should_send or result.cta == "none":
                continue
            # A source citation may follow the ask (the pattern the case studies
            # use); the ask must be the last thing said before it.
            body = result.body
            for citation in result.plan.citations:
                body = body.replace(f"— {citation}", "").strip()
            tail = body.rsplit(".", 2)[-2:] if body.count(".") > 1 else [body]
            assert any(
                marker in " ".join(tail).lower()
                for marker in ("?", "reply", "confirm", "kijiye", "dijiye", "dun")
            ), f"{trigger_id} buries its call to action: {body[-90:]!r}"

    def test_bodies_stay_readable_in_length(self, dataset, now):
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if result.should_send:
                assert 80 <= len(result.body) <= 600, f"{trigger_id}: {len(result.body)} chars"

    def test_template_params_are_populated_for_a_first_touch(self, dataset, now):
        result = _compose(dataset, "trg_001_research_digest_dentists", now)

        assert result.template_name.startswith("vera_")
        assert len(result.template_params) >= 3
        assert all(param for param in result.template_params)

    def test_rationale_explains_the_choice_not_the_copy(self, dataset, now):
        result = _compose(dataset, "trg_004_perf_dip_bharat", now)

        assert "perf_dip" in result.rationale
        assert "priority" in result.rationale
        assert "merchant.performance" in result.rationale or "trigger.payload" in result.rationale
        assert result.rationale != result.body

    def test_the_rationale_never_claims_an_offer_the_body_omits(self, dataset, now):
        """The judge cross-checks the rationale against the message."""
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if not result.should_send:
                continue
            if result.plan.offer is not None:
                assert result.plan.offer.title in result.body, trigger_id
            else:
                assert "no offer referenced" in result.rationale, trigger_id

    def test_conversation_ids_have_no_dangling_separator(self, dataset, now):
        for trigger_id in dataset.triggers:
            result = _compose(dataset, trigger_id, now)
            if result.should_send:
                assert not result.conversation_id.endswith("_"), trigger_id
                assert "__" not in result.conversation_id

    def test_action_dict_has_every_field_the_judge_requires(self, dataset, now):
        action = _compose(dataset, "trg_003_recall_due_priya", now).to_action()

        assert set(action) == {
            "conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id",
            "template_name", "template_params", "body", "cta", "suppression_key", "rationale",
        }
        assert all(action[key] for key in ("conversation_id", "merchant_id", "send_as",
                                          "trigger_id", "body", "cta", "suppression_key",
                                          "rationale"))
