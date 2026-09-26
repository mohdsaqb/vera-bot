"""Normalization against the real challenge schema.

These tests read the seed files, so they fail if the engine drifts from the
actual field names rather than from a guess about them.
"""

from __future__ import annotations

from conftest import variant
from engine.normalize import (
    days_between,
    infer_offer_type,
    normalize_category,
    normalize_customer,
    normalize_merchant,
    normalize_trigger,
    parse_iso,
    parse_signal_values,
)


class TestCategory:
    def test_reads_every_supplied_block(self, dataset):
        category = normalize_category(dataset.categories["dentists"])

        assert category.slug == "dentists"
        assert category.display_name == "Dentists"
        assert category.voice.tone == "peer_clinical"
        assert "fluoride varnish" in category.voice.vocab_allowed
        assert "guaranteed" in category.voice.vocab_taboo
        assert len(category.offer_catalog) == 8
        assert len(category.digest) == 5
        assert category.peer("avg_ctr") == 0.030
        assert category.seasonal_beats and category.trend_signals
        assert "Dental Council of India (DCI)" in category.regulators

    def test_digest_items_resolve_by_id(self, dataset):
        category = normalize_category(dataset.categories["dentists"])
        item = category.digest_item("d_2026W17_jida_fluoride")

        assert item is not None
        assert item.kind == "research"
        assert item.source == "JIDA Oct 2026, p.14"
        assert item.extra["trial_n"] == 2100
        assert item.extra["patient_segment"] == "high_risk_adults"
        assert category.digest_item("does_not_exist") is None
        assert category.digest_item(None) is None

    def test_all_five_categories_normalize(self, dataset):
        for slug, payload in dataset.categories.items():
            category = normalize_category(payload)
            assert category is not None
            assert category.slug == slug
            assert category.voice.tone
            assert category.offer_catalog

    def test_empty_and_none_payloads(self):
        assert normalize_category(None) is None
        assert normalize_category({}) is None


class TestMerchant:
    def test_reads_the_seed_merchant(self, dataset):
        merchant = normalize_merchant(dataset.merchants["m_001_drmeera_dentist_delhi"])

        assert merchant.name == "Dr. Meera's Dental Clinic"
        assert merchant.owner_first_name == "Meera"
        assert merchant.locality == "Lajpat Nagar"
        assert merchant.verified is True
        assert merchant.languages == ("en", "hi")
        assert merchant.metric("views") == 2410
        assert merchant.delta("calls") == -0.05
        assert merchant.aggregate("high_risk_adult_count") == 124
        assert merchant.has_signal("ctr_below_peer_median")
        assert merchant.signal_values["stale_posts"] == 22
        assert merchant.has_talked_before is True

    def test_active_and_expired_offers_are_separated(self, dataset):
        merchant = normalize_merchant(dataset.merchants["m_001_drmeera_dentist_delhi"])

        assert [o.title for o in merchant.active_offers] == ["Dental Cleaning @ ₹299"]
        assert [o.title for o in merchant.expired_offers] == ["Deep Cleaning @ ₹499"]

    def test_generated_shape_with_empty_optional_blocks(self, dataset):
        # Generated merchants carry no offers, signals, review themes or history.
        payload = variant(
            dataset.merchants["m_001_drmeera_dentist_delhi"],
            offers=[], signals=[], review_themes=[], conversation_history=[],
        )
        merchant = normalize_merchant(payload)

        assert merchant.offers == ()
        assert merchant.signals == ()
        assert merchant.signal_values == {}
        assert merchant.has_talked_before is False
        assert merchant.metric("views") == 2410

    def test_missing_nonessential_fields_do_not_raise(self):
        merchant = normalize_merchant({"merchant_id": "m_x", "category_slug": "salons"})

        assert merchant is not None
        assert merchant.name == ""
        assert merchant.metric("views") is None
        assert merchant.delta("calls") is None
        assert merchant.verified is None
        assert merchant.active_offers == ()

    def test_all_ten_seed_merchants_normalize(self, dataset):
        for merchant_id, payload in dataset.merchants.items():
            merchant = normalize_merchant(payload)
            assert merchant.merchant_id == merchant_id
            assert merchant.category_slug in dataset.categories


class TestCustomer:
    def test_reads_the_seed_customer(self, dataset):
        customer = normalize_customer(dataset.customers["c_001_priya_for_m001"])

        assert customer.name == "Priya"
        assert customer.language_pref == "hi-en mix"
        assert customer.state == "lapsed_soft"
        assert customer.visits_total == 4
        assert customer.services_received == ("cleaning", "cleaning", "whitening", "cleaning")
        assert customer.preferred_slots == "weekday_evening"
        assert customer.consent_scope == ("recall_reminders", "appointment_reminders")
        assert customer.has_any_consent is True
        assert customer.contactable is True

    def test_senior_flag_and_extra_relationship_fields(self, dataset):
        customer = normalize_customer(dataset.customers["c_013_grandfather_for_m009"])

        assert customer.is_senior is True
        assert customer.relationship_extra["chronic_conditions"]
        assert customer.channel == "whatsapp_via_son"

    def test_walk_in_without_phone_or_channel_is_not_contactable(self, dataset):
        customer = normalize_customer(dataset.customers["c_015_anonymous_for_m010"])

        assert customer.contactable is False
        assert customer.has_any_consent is False
        assert customer.reminder_opt_in is False

    def test_empty_payload(self):
        assert normalize_customer(None) is None
        assert normalize_customer({}) is None


class TestTrigger:
    def test_reads_the_seed_trigger(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_003_recall_due_priya"])

        assert trigger.kind == "recall_due"
        assert trigger.scope == "customer"
        assert trigger.source == "internal"
        assert trigger.urgency == 3
        assert trigger.customer_id == "c_001_priya_for_m001"
        assert trigger.suppression_key == "recall:c_001_priya_for_m001:6mo"
        assert trigger.payload["service_due"] == "6_month_cleaning"
        assert trigger.is_placeholder is False

    def test_null_customer_id_becomes_none(self, dataset):
        trigger = normalize_trigger(dataset.triggers["trg_001_research_digest_dentists"])
        assert trigger.customer_id is None

    def test_generated_placeholder_payload_is_flagged(self):
        trigger = normalize_trigger(
            {"id": "trg_x", "kind": "perf_dip", "scope": "merchant", "urgency": 3,
             "payload": {"placeholder": True, "metric_or_topic": "perf_dip"}}
        )
        assert trigger.is_placeholder is True

    def test_urgency_is_clamped_and_defaulted(self):
        assert normalize_trigger({"id": "t", "kind": "k", "urgency": 9}).urgency == 5
        assert normalize_trigger({"id": "t", "kind": "k", "urgency": 0}).urgency == 1
        assert normalize_trigger({"id": "t", "kind": "k"}).urgency == 1

    def test_all_seed_triggers_normalize(self, dataset):
        for trigger_id, payload in dataset.triggers.items():
            trigger = normalize_trigger(payload)
            assert trigger.id == trigger_id
            assert trigger.scope in {"merchant", "customer"}
            assert trigger.source in {"internal", "external"}


class TestHelpers:
    def test_parse_iso_handles_every_shape_in_the_dataset(self):
        assert parse_iso("2026-05-03T00:00:00Z") is not None
        assert parse_iso("2026-04-28T00:00:00+05:30") is not None
        assert parse_iso("2026-11-12") is not None
        assert parse_iso("not a date") is None
        assert parse_iso(None) is None
        assert parse_iso("") is None

    def test_parse_iso_returns_utc(self):
        moment = parse_iso("2026-04-26T10:00:00+05:30")
        assert moment.tzinfo is not None
        assert moment.hour == 4 and moment.minute == 30

    def test_days_between(self):
        assert days_between("2026-05-12", "2026-11-12") == 184
        assert days_between("2026-11-12", "2026-05-12") == -184
        assert days_between("junk", "2026-05-12") is None

    def test_signal_values_decodes_both_encodings(self):
        values = parse_signal_values(
            ("stale_posts:22d", "dormant_with_vera_14d", "ctr_below_peer_median")
        )
        assert values == {
            "stale_posts": 22, "dormant_with_vera": 14, "ctr_below_peer_median": True,
        }

    def test_offer_type_inference(self):
        assert infer_offer_type("Dental Cleaning @ ₹299") == "service_at_price"
        assert infer_offer_type("Flat 20% OFF on medicines") == "percentage_discount"
        assert infer_offer_type("Buy 1 Pizza Get 1 Free (Tue-Thu)") == "bogo"
        assert infer_offer_type("Free Body Composition Analysis") == "free_service"
        assert infer_offer_type("anything", declared="membership") == "membership"
