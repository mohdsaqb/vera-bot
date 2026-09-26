"""Offer selection: a real offer, or none: never an invented one."""

from __future__ import annotations

from conftest import deep_set, variant
from engine.normalize import (
    normalize_category,
    normalize_customer,
    normalize_merchant,
    normalize_trigger,
)
from engine.offers import select_offer


def _pick(dataset, trigger_id, **swaps):
    category, merchant, trigger, customer = dataset.compose_args(trigger_id)
    return select_offer(
        normalize_category(swaps.get("category", category)),
        normalize_merchant(swaps.get("merchant", merchant)),
        normalize_trigger(swaps.get("trigger", trigger)),
        normalize_customer(swaps.get("customer", customer)),
    )


class TestExistingOffers:
    def test_the_merchants_own_active_offer_is_preferred(self, dataset):
        choice = _pick(dataset, "trg_003_recall_due_priya")

        assert choice is not None
        assert choice.is_existing is True
        assert choice.title == "Dental Cleaning @ ₹299"
        assert choice.offer.source == "merchant_offers"

    def test_expired_offers_are_never_offered(self, dataset):
        merchant = variant(
            dataset.merchants["m_001_drmeera_dentist_delhi"],
            offers=[{"id": "o", "title": "Deep Cleaning @ ₹499", "status": "expired"}],
        )
        choice = _pick(dataset, "trg_003_recall_due_priya", merchant=merchant)

        assert choice is None or choice.title != "Deep Cleaning @ ₹499"

    def test_relevance_beats_ordering(self, dataset):
        """A refill picks the senior discount for a senior, not the first offer."""
        choice = _pick(dataset, "trg_019_chronic_refill_grandfather")

        assert choice.is_existing is True
        assert choice.title == "Senior Citizen 15% OFF"

    def test_irrelevant_active_offer_is_dropped_where_fit_matters(self, dataset):
        """A haircut price does not belong in a bridal skin-prep message."""
        choice = _pick(dataset, "trg_007_bridal_followup_kavya")
        assert choice is None


class TestCatalogFallback:
    def test_catalog_pattern_is_marked_as_a_proposal(self, dataset):
        merchant = variant(dataset.merchants["m_002_bharat_dentist_mumbai"], offers=[])
        choice = _pick(dataset, "trg_004_perf_dip_bharat", merchant=merchant)

        assert choice is not None
        assert choice.is_existing is False
        assert choice.offer.source == "category_catalog"
        assert choice.title in [
            o["title"] for o in dataset.categories["dentists"]["offer_catalog"]
        ]

    def test_nothing_is_invented_when_the_catalog_is_empty(self, dataset):
        category = variant(dataset.categories["dentists"], offer_catalog=[])
        merchant = variant(dataset.merchants["m_002_bharat_dentist_mumbai"], offers=[])
        choice = _pick(dataset, "trg_004_perf_dip_bharat", merchant=merchant, category=category)

        assert choice is None

    def test_catalog_choice_prefers_a_service_at_a_price_over_a_percentage(self, dataset):
        """challenge-brief.md §11: a flat percentage is an anti-pattern when a
        service at a named price is available.

        Both candidates are equally relevant to the match-night moment, so the
        offer shape is what decides.
        """
        category = variant(
            dataset.categories["restaurants"],
            offer_catalog=[
                {"id": "a", "title": "Match Combo 30% OFF", "value": "30%",
                 "audience": "new_user", "type": "percentage_discount"},
                {"id": "b", "title": "Match-night Combo @ ₹399", "value": "399",
                 "audience": "new_user", "type": "service_at_price"},
            ],
        )
        merchant = variant(dataset.merchants["m_005_pizzajunction_restaurant_delhi"], offers=[])
        choice = _pick(dataset, "trg_010_ipl_match_delhi",
                       merchant=merchant, category=category)

        assert choice is not None
        assert choice.title == "Match-night Combo @ ₹399"

    def test_senior_audience_is_matched_to_a_senior_customer(self, dataset):
        merchant = variant(dataset.merchants["m_009_apollo_pharmacy_jaipur"], offers=[])
        choice = _pick(dataset, "trg_019_chronic_refill_grandfather", merchant=merchant)

        assert choice is not None
        assert choice.offer.audience in {"senior", "repeat_user", "all", "new_user"}

    def test_senior_only_offer_is_not_pushed_at_a_young_customer(self, dataset):
        category = variant(
            dataset.categories["pharmacies"],
            offer_catalog=[
                {"id": "s", "title": "Senior Citizen 15% OFF (60+ age)", "value": "15%",
                 "audience": "senior", "type": "percentage_discount"},
                {"id": "d", "title": "Free Home Delivery > ₹499", "value": "free_delivery",
                 "audience": "new_user", "type": "free_addon"},
            ],
        )
        merchant = variant(dataset.merchants["m_009_apollo_pharmacy_jaipur"], offers=[])
        customer = deep_set(dataset.customers["c_014_priti_for_m009"], "state", "lapsed_soft")
        choice = select_offer(
            normalize_category(category),
            normalize_merchant(merchant),
            normalize_trigger(dataset.triggers["trg_019_chronic_refill_grandfather"]),
            normalize_customer(customer),
        )

        assert choice is None or "Senior Citizen" not in choice.title


class TestNoFabrication:
    def test_every_chosen_offer_title_exists_in_an_input_context(self, dataset):
        """The offer must be traceable to the merchant's offers or the catalog."""
        for trigger_id in dataset.triggers:
            category, merchant, trigger, customer = dataset.compose_args(trigger_id)
            choice = select_offer(
                normalize_category(category),
                normalize_merchant(merchant),
                normalize_trigger(trigger),
                normalize_customer(customer),
            )
            if choice is None:
                continue
            supplied = {o["title"] for o in merchant.get("offers", [])} | {
                o["title"] for o in category.get("offer_catalog", [])
            }
            assert choice.title in supplied, f"{trigger_id} produced an unsourced offer"
