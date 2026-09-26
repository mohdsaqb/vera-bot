"""Trigger evaluation and ranking: should Vera speak, and about what first."""

from __future__ import annotations

from conftest import deep_set, variant
from engine import policy
from engine.normalize import (
    normalize_category,
    normalize_customer,
    normalize_merchant,
    normalize_trigger,
)
from engine.triggers import (
    category_fit_score,
    consent_check,
    customer_state_check,
    evaluate_trigger,
    merchant_relevance_score,
    rank_triggers,
    time_pressure_score,
    urgency_score,
)


def _evaluate(dataset, trigger_id, now=None, **swaps):
    category, merchant, trigger, customer = dataset.compose_args(trigger_id)
    return evaluate_trigger(
        normalize_category(swaps.get("category", category)),
        normalize_merchant(swaps.get("merchant", merchant)),
        normalize_trigger(swaps.get("trigger", trigger)),
        normalize_customer(swaps.get("customer", customer)),
        now,
    )


class TestComponents:
    def test_urgency_maps_to_a_unit_range(self):
        assert urgency_score(normalize_trigger({"id": "t", "kind": "k", "urgency": 1})) == 0.0
        assert urgency_score(normalize_trigger({"id": "t", "kind": "k", "urgency": 5})) == 1.0

    def test_category_restriction(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_019_chronic_refill_grandfather"])
        pharmacy = normalize_merchant(dataset.merchants["m_009_apollo_pharmacy_jaipur"])
        dentist = normalize_merchant(dataset.merchants["m_001_drmeera_dentist_delhi"])

        assert category_fit_score(trigger, pharmacy) == 1.0
        assert category_fit_score(trigger, dentist) == 0.0

    def test_payload_declared_relevance_wins_over_the_table(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_006_festival_diwali"])
        salon = normalize_merchant(dataset.merchants["m_003_studio11_salon_hyderabad"])
        gym = normalize_merchant(dataset.merchants["m_007_powerhouse_gym_bangalore"])

        # The Diwali payload lists salons/restaurants/pharmacies, not gyms.
        assert category_fit_score(trigger, salon) == 1.0
        assert category_fit_score(trigger, gym) == 0.0

    def test_relevance_is_zero_when_the_merchant_contradicts_the_trigger(self, dataset):
        category = normalize_category(dataset.categories["dentists"])
        trigger = normalize_trigger(
            variant(dataset.triggers["trg_004_perf_dip_bharat"],
                    payload={"metric": "calls", "window": "7d"})
        )
        rising = normalize_merchant(
            deep_set(dataset.merchants["m_002_bharat_dentist_mumbai"],
                     "performance.delta_7d.calls_pct", 0.3)
        )
        score, note = merchant_relevance_score(trigger, rising, category)

        assert score == 0.0
        assert "no material" in note

    def test_verified_listing_contradicts_an_unverified_trigger(self, dataset):
        category = normalize_category(dataset.categories["pharmacies"])
        trigger = normalize_trigger(dataset.triggers["trg_021_unverified_gbp_sunrise"])
        verified = normalize_merchant(
            deep_set(dataset.merchants["m_010_sunrisepharm_pharmacy_lucknow"],
                     "identity.verified", True)
        )
        assert merchant_relevance_score(trigger, verified, category)[0] == 0.0

    def test_time_pressure_is_neutral_without_a_reference_time(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_010_ipl_match_delhi"])
        score, note = time_pressure_score(trigger, None)

        assert score == 0.4
        assert "no reference time" in note

    def test_time_pressure_rises_as_the_window_closes(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_002_compliance_dci_radiograph"])
        far = time_pressure_score(trigger, "2026-04-26T00:00:00Z")[0]
        near = time_pressure_score(trigger, "2026-12-14T00:00:00Z")[0]

        assert near > far


class TestHardBlocks:
    def test_expired_trigger(self, dataset):
        evaluation = _evaluate(dataset, "trg_010_ipl_match_delhi", now="2026-05-01T00:00:00Z")

        assert evaluation.eligible is False
        assert evaluation.reason_code == policy.REASON_EXPIRED

    def test_not_expired_before_the_deadline(self, dataset):
        evaluation = _evaluate(dataset, "trg_010_ipl_match_delhi", now="2026-04-26T10:00:00Z")
        assert evaluation.eligible is True

    def test_category_mismatch(self, dataset):
        evaluation = _evaluate(
            dataset, "trg_019_chronic_refill_grandfather",
            merchant=dataset.merchants["m_001_drmeera_dentist_delhi"],
        )
        assert evaluation.reason_code == policy.REASON_CATEGORY_MISMATCH

    def test_distant_dated_event(self, dataset):
        evaluation = _evaluate(dataset, "trg_006_festival_diwali", now="2026-04-26T00:00:00Z")

        assert evaluation.eligible is False
        assert evaluation.reason_code == policy.REASON_EVENT_DISTANT

    def test_open_preparation_window_beats_the_lead_limit(self, dataset):
        # Kavya's wedding is 196 days out, but the payload says the 30-day
        # skin-prep window is open now.
        evaluation = _evaluate(dataset, "trg_007_bridal_followup_kavya", now="2026-04-26T00:00:00Z")
        assert evaluation.eligible is True

    def test_missing_contexts(self, dataset):
        _, merchant, trigger, _ = dataset.compose_args("trg_001_research_digest_dentists")

        assert evaluate_trigger(
            None, normalize_merchant(merchant), normalize_trigger(trigger)
        ).reason_code == policy.REASON_MISSING_CONTEXT
        assert evaluate_trigger(
            normalize_category(dataset.categories["dentists"]), None, normalize_trigger(trigger)
        ).reason_code == policy.REASON_MISSING_CONTEXT
        assert evaluate_trigger(None, None, None).reason_code == policy.REASON_MISSING_CONTEXT

    def test_low_priority_is_not_eligible_but_keeps_its_score(self, dataset):
        category, merchant, trigger, _ = dataset.compose_args("trg_012_milestone_mylari")
        weak = normalize_trigger(variant(trigger, urgency=1, payload={"metric": "review_count"}))
        evaluation = evaluate_trigger(
            normalize_category(category), normalize_merchant(merchant), weak, None
        )
        assert evaluation.eligible is False


class TestConsent:
    def test_merchant_scoped_triggers_need_no_consent(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_001_research_digest_dentists"])
        assert consent_check(trigger, None) == (True, "", "")

    def test_matching_purpose_is_allowed(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_003_recall_due_priya"])
        customer = normalize_customer(dataset.customers["c_001_priya_for_m001"])
        allowed, code, note = consent_check(trigger, customer)

        assert allowed is True
        assert "recall_reminders" in note

    def test_promotional_consent_does_not_authorise_a_clinical_reminder(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_003_recall_due_priya"])
        customer = normalize_customer(
            deep_set(dataset.customers["c_001_priya_for_m001"],
                     "consent.scope", ["promotional_offers"])
        )
        allowed, code, _ = consent_check(trigger, customer)

        assert allowed is False
        assert code == policy.REASON_NO_CONSENT

    def test_promotional_consent_does_authorise_a_winback(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_015_winback_rashmi"])
        customer = normalize_customer(
            deep_set(dataset.customers["c_010_rashmi_for_m007"],
                     "consent.scope", ["promotional_offers"])
        )
        assert consent_check(trigger, customer)[0] is True

    def test_explicit_opt_out_blocks_everything(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_003_recall_due_priya"])
        customer = normalize_customer(
            deep_set(dataset.customers["c_001_priya_for_m001"],
                     "preferences.reminder_opt_in", False)
        )
        assert consent_check(trigger, customer)[1] == policy.REASON_OPTED_OUT

    def test_unreachable_customer_is_blocked(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_019_chronic_refill_grandfather"])
        customer = normalize_customer(dataset.customers["c_015_anonymous_for_m010"])
        assert consent_check(trigger, customer)[1] == policy.REASON_NOT_CONTACTABLE

    def test_customer_scoped_trigger_without_a_customer(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_003_recall_due_priya"])
        assert consent_check(trigger, None)[1] == policy.REASON_CUSTOMER_MISSING


class TestCustomerState:
    def test_winback_needs_a_lapsed_customer(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_015_winback_rashmi"])
        lapsed = normalize_customer(dataset.customers["c_010_rashmi_for_m007"])
        active = normalize_customer(
            deep_set(dataset.customers["c_010_rashmi_for_m007"], "state", "active")
        )

        assert customer_state_check(trigger, lapsed)[0] is True
        allowed, note = customer_state_check(trigger, active)
        assert allowed is False
        assert "active" in note

    def test_trial_followup_needs_a_new_customer(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_017_kids_yoga_trial_followup_karthik"])
        new = normalize_customer(dataset.customers["c_012_karthik_jr_for_m008"])
        veteran = normalize_customer(dataset.customers["c_011_sumitra_for_m008"])

        assert customer_state_check(trigger, new)[0] is True
        assert customer_state_check(trigger, veteran)[0] is False

    def test_families_without_a_state_rule_are_unaffected(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_001_research_digest_dentists"])
        assert customer_state_check(trigger, None)[0] is True


class TestRanking:
    def test_higher_priority_comes_first(self, dataset):
        ids = [
            "trg_012_milestone_mylari",              # urgency 1
            "trg_018_supply_atorvastatin_recall",    # urgency 5
            "trg_004_perf_dip_bharat",               # urgency 4
        ]
        evaluations, triggers = [], {}
        for trigger_id in ids:
            category, merchant, trigger, customer = dataset.compose_args(trigger_id)
            normalized = normalize_trigger(trigger)
            triggers[normalized.id] = normalized
            evaluations.append(
                evaluate_trigger(
                    normalize_category(category), normalize_merchant(merchant),
                    normalized, normalize_customer(customer), "2026-04-26T10:35:00Z",
                )
            )
        ranked = rank_triggers(evaluations, triggers)

        assert ranked[0].trigger_id == "trg_018_supply_atorvastatin_recall"
        assert [r.priority for r in ranked] == sorted(
            (r.priority for r in ranked), reverse=True
        )

    def test_ineligible_triggers_are_dropped(self, dataset):
        category, merchant, trigger, _ = dataset.compose_args("trg_006_festival_diwali")
        normalized = normalize_trigger(trigger)
        evaluation = evaluate_trigger(
            normalize_category(category), normalize_merchant(merchant), normalized,
            None, "2026-04-26T00:00:00Z",
        )
        assert rank_triggers([evaluation], {normalized.id: normalized}) == []

    def test_ranking_is_order_independent(self, dataset):
        ids = ["trg_001_research_digest_dentists", "trg_002_compliance_dci_radiograph",
               "trg_022_cde_webinar_dentists", "trg_023_competitor_opened_dentist"]
        evaluations, triggers = [], {}
        for trigger_id in ids:
            category, merchant, trigger, customer = dataset.compose_args(trigger_id)
            normalized = normalize_trigger(trigger)
            triggers[normalized.id] = normalized
            evaluations.append(
                evaluate_trigger(
                    normalize_category(category), normalize_merchant(merchant),
                    normalized, None, "2026-04-26T10:35:00Z",
                )
            )
        forward = [r.trigger_id for r in rank_triggers(evaluations, triggers)]
        backward = [r.trigger_id for r in rank_triggers(list(reversed(evaluations)), triggers)]

        assert forward == backward
