"""Signal extraction: which single fact the engine chooses, and how it is worded."""

from __future__ import annotations

from conftest import deep_set, variant
from engine import policy
from engine.normalize import (
    normalize_category,
    normalize_customer,
    normalize_merchant,
    normalize_trigger,
)
from engine.signals import (
    clean_service_label,
    is_material_delta,
    offer_gap_signal,
    peer_gap_signal,
    performance_delta_signal,
    review_theme_signal,
    select_primary_signal,
    select_supporting_fact,
    strongest_delta_signal,
    subscription_signal,
)


def _norm(dataset, trigger_id):
    category, merchant, trigger, customer = dataset.compose_args(trigger_id)
    return (
        normalize_category(category),
        normalize_merchant(merchant),
        normalize_trigger(trigger),
        normalize_customer(customer),
    )


class TestMateriality:
    def test_small_swings_are_not_material(self, dataset):
        merchant = normalize_merchant(
            deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"],
                     "performance.delta_7d.calls_pct", -0.05)
        )
        assert is_material_delta(merchant, "calls") is False
        assert performance_delta_signal(merchant, "calls") is None

    def test_large_swings_on_a_real_base_are_material(self, dataset):
        merchant = normalize_merchant(
            deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"],
                     "performance.delta_7d.calls_pct", -0.4)
        )
        assert is_material_delta(merchant, "calls") is True

    def test_percentage_on_a_tiny_base_is_rejected(self, dataset):
        payload = deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"],
                           "performance.calls", 3)
        payload = deep_set(payload, "performance.delta_7d.calls_pct", 0.5)
        merchant = normalize_merchant(payload)

        # +50% of 3 calls is one or two calls: a percentage, not a story.
        assert is_material_delta(merchant, "calls") is False

    def test_direction_filter(self, dataset):
        merchant = normalize_merchant(
            deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"],
                     "performance.delta_7d.views_pct", 0.30)
        )
        assert performance_delta_signal(merchant, "views", want="up") is not None
        assert performance_delta_signal(merchant, "views", want="down") is None

    def test_strongest_delta_picks_the_biggest_move(self, dataset):
        payload = deep_set(dataset.merchants["m_003_studio11_salon_hyderabad"],
                           "performance.delta_7d.views_pct", 0.18)
        payload = deep_set(payload, "performance.delta_7d.calls_pct", 0.45)
        merchant = normalize_merchant(payload)

        signal = strongest_delta_signal(merchant, want="up")
        assert signal.data["metric"] == "calls"
        assert "45%" in signal.text


class TestWording:
    def test_delta_is_phrased_with_a_percentage_and_a_base(self, dataset):
        merchant = normalize_merchant(dataset.merchants["m_007_powerhouse_gym_bangalore"])
        signal = performance_delta_signal(merchant, "views", want="down")

        assert "profile views are down 30% week-on-week" in signal.text
        assert "1,480" in signal.text
        assert signal.source == "merchant.performance.delta_7d.views_pct"

    def test_ctr_gap_is_expressed_in_percentage_points(self, dataset):
        category, merchant, _, _ = _norm(dataset, "trg_001_research_digest_dentists")
        signal = peer_gap_signal(merchant, category, want="below")

        assert "2.1%" in signal.text and "3%" in signal.text
        assert "0.9 percentage points below" in signal.text
        assert signal.data["side"] == "below"

    def test_above_peer_is_reported_as_above(self, dataset):
        category, merchant, _, _ = _norm(dataset, "trg_024_perf_spike_zen")
        signal = peer_gap_signal(merchant, category)

        assert signal.data["side"] == "above"
        assert "above" in signal.text

    def test_no_gap_when_the_merchant_sits_on_the_median(self, dataset):
        category = normalize_category(dataset.categories["dentists"])
        merchant = normalize_merchant(
            deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"], "performance.ctr", 0.030)
        )
        assert peer_gap_signal(merchant, category, want="below") is None

    def test_service_labels_are_cleaned(self):
        assert clean_service_label("membership_x4") == "membership"
        assert clean_service_label("chronic_rx_metformin") == "metformin"
        assert clean_service_label("weekday_thali") == "weekday thali"


class TestStateSignals:
    def test_offer_gap_names_the_lapsed_offer(self, dataset):
        merchant = normalize_merchant(
            variant(dataset.merchants["m_001_drmeera_dentist_delhi"],
                    offers=[{"id": "o1", "title": "Deep Cleaning @ ₹499", "status": "expired"}])
        )
        signal = offer_gap_signal(merchant)
        assert "Deep Cleaning @ ₹499" in signal.text

    def test_no_offer_gap_when_one_is_live(self, dataset):
        merchant = normalize_merchant(dataset.merchants["m_001_drmeera_dentist_delhi"])
        assert offer_gap_signal(merchant) is None

    def test_expired_subscription(self, dataset):
        merchant = normalize_merchant(dataset.merchants["m_004_glamour_salon_pune"])
        signal = subscription_signal(merchant)
        assert "38 days ago" in signal.text

    def test_subscription_far_from_renewal_is_not_a_signal(self, dataset):
        merchant = normalize_merchant(
            dataset.merchants["m_006_southindiancafe_restaurant_bangalore"]
        )
        assert subscription_signal(merchant) is None

    def test_review_theme_picks_the_most_repeated(self, dataset):
        merchant = normalize_merchant(dataset.merchants["m_005_pizzajunction_restaurant_delhi"])
        negative = review_theme_signal(merchant, sentiment="neg")
        positive = review_theme_signal(merchant, sentiment="pos")

        assert "delivery late" in negative.text and "4 reviews" in negative.text
        assert "pizza quality" in positive.text

    def test_no_review_signal_when_none_recorded(self, dataset):
        merchant = normalize_merchant(dataset.merchants["m_010_sunrisepharm_pharmacy_lucknow"])
        assert review_theme_signal(merchant) is None


class TestPrimarySelection:
    def test_trigger_payload_beats_context_when_it_has_content(self, dataset):
        category, merchant, trigger, customer = _norm(dataset, "trg_004_perf_dip_bharat")
        signal = select_primary_signal(category, merchant, trigger, customer)

        assert signal.source == "trigger.payload.delta_pct"
        assert "down 50%" in signal.text
        assert "baseline of 12" in signal.text

    def test_research_digest_resolves_the_item_and_keeps_the_citation(self, dataset):
        category, merchant, trigger, customer = _norm(dataset, "trg_001_research_digest_dentists")
        signal = select_primary_signal(category, merchant, trigger, customer)

        assert signal.citation == "JIDA Oct 2026, p.14"
        assert "38% lower caries recurrence" in signal.text
        assert "n=2,100" in signal.text
        assert signal.source.startswith("category.digest")

    def test_unresolvable_digest_reference_yields_no_signal(self, dataset):
        category, merchant, trigger, _ = _norm(dataset, "trg_001_research_digest_dentists")
        broken = normalize_trigger(
            variant(trigger.raw, payload={"category": "dentists", "top_item_id": "nope"})
        )
        assert select_primary_signal(category, merchant, broken, None) is None

    def test_placeholder_payload_falls_back_to_merchant_data(self, dataset):
        category, merchant, trigger, _ = _norm(dataset, "trg_004_perf_dip_bharat")
        placeholder = normalize_trigger(
            variant(trigger.raw, payload={"placeholder": True, "metric_or_topic": "perf_dip"})
        )
        signal = select_primary_signal(category, merchant, placeholder, None)

        assert signal is not None
        assert signal.source.startswith("merchant.")

    def test_customer_facing_never_uses_merchant_analytics(self, dataset):
        """A customer must not be told the shop's views or click-through rate."""
        category, merchant, trigger, customer = _norm(dataset, "trg_003_recall_due_priya")
        placeholder = normalize_trigger(
            variant(trigger.raw, payload={"placeholder": True, "metric_or_topic": "recall_due"})
        )
        signal = select_primary_signal(category, merchant, placeholder, customer)

        assert signal is None or signal.kind in policy.CUSTOMER_FACT_KINDS
        if signal is not None:
            assert not signal.source.startswith("merchant.")


class TestSupportingFact:
    def test_research_item_is_anchored_on_the_matching_cohort(self, dataset):
        category, merchant, trigger, _ = _norm(dataset, "trg_001_research_digest_dentists")
        primary = select_primary_signal(category, merchant, trigger, None)
        supporting = select_supporting_fact(category, merchant, trigger, None, primary)

        # The digest item names high_risk_adults; the merchant records 124 of them.
        assert supporting.kind == "cohort_match"
        assert "124" in supporting.text
        assert "high-risk adults" in supporting.text

    def test_cohort_pairing_needs_both_halves(self, dataset):
        category, merchant, trigger, _ = _norm(dataset, "trg_001_research_digest_dentists")
        without_count = normalize_merchant(
            deep_set(dataset.merchants["m_001_drmeera_dentist_delhi"],
                     "customer_aggregate", {"total_unique_ytd": 540})
        )
        primary = select_primary_signal(category, without_count, trigger, None)
        supporting = select_supporting_fact(category, without_count, trigger, None, primary)

        assert supporting is None or supporting.kind != "cohort_match"

    def test_supporting_fact_never_duplicates_the_primary(self, dataset):
        category, merchant, trigger, customer = _norm(dataset, "trg_019_chronic_refill_grandfather")
        primary = select_primary_signal(category, merchant, trigger, customer)
        supporting = select_supporting_fact(category, merchant, trigger, customer, primary)

        assert supporting is not None
        assert supporting.kind != primary.kind
        assert "metformin" not in supporting.text
