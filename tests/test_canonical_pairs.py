"""The 30 canonical (merchant, trigger) pairs from the expanded dataset.

`challenge-brief.md` §6 defines a fixed submission test set that every
participant produces a message for. These tests run the engine over that set and
assert *general* properties: no expected body is hardcoded, because pinning the
answers would tune the engine to the cases instead of improving the rules.

Skipped when `expanded/` has not been generated:
    python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from conftest import NOW, challenge_dir
from engine import compose
from engine.render import validate_body

# The engine must justify, for every pair, either a message or its silence.
VALID_REASONS = {
    "sent",
    "trigger_expired",
    "trigger_not_relevant_to_category",
    "trigger_claim_not_supported_by_data",
    "no_verifiable_fact_available",
    "customer_consent_missing_for_purpose",
    "customer_opted_out_of_reminders",
    "customer_not_contactable",
    "required_context_missing",
    "event_too_distant_to_act_on",
    "not_worth_interrupting",
    "customer_context_required_but_absent",
}


@lru_cache(maxsize=1)
def _expanded() -> dict[str, Any] | None:
    root = challenge_dir() / "expanded"
    pairs_file = root / "test_pairs.json"
    if not pairs_file.is_file():
        return None

    def index(folder: str, key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for path in sorted((root / folder).glob("*.json")):
            payload = json.loads(path.read_text())
            out[payload.get(key, path.stem)] = payload
        return out

    return {
        "pairs": json.loads(pairs_file.read_text())["pairs"],
        "categories": index("categories", "slug"),
        "merchants": index("merchants", "merchant_id"),
        "customers": index("customers", "customer_id"),
        "triggers": index("triggers", "id"),
    }


@pytest.fixture(scope="module")
def expanded() -> dict[str, Any]:
    data = _expanded()
    if data is None:
        pytest.skip("expanded dataset not generated; see this module's docstring")
    return data


@pytest.fixture(scope="module")
def outcomes(expanded) -> list[tuple[dict[str, Any], Any]]:
    """`compose()` run once over all 30 pairs, as (pair, result)."""
    results = []
    for pair in expanded["pairs"]:
        merchant = expanded["merchants"].get(pair["merchant_id"])
        trigger = expanded["triggers"].get(pair["trigger_id"])
        customer = expanded["customers"].get(pair.get("customer_id") or "")
        category = expanded["categories"].get((merchant or {}).get("category_slug", ""))
        results.append((pair, compose(category, merchant, trigger, customer, now=NOW)))
    return results


class TestCoverage:
    def test_the_set_is_thirty_pairs(self, expanded):
        assert len(expanded["pairs"]) == 30

    def test_every_pair_reaches_a_justified_decision(self, outcomes):
        for pair, result in outcomes:
            assert result.decision in {"send", "no_action"}, pair["test_id"]
            assert result.reason_code in VALID_REASONS, (pair["test_id"], result.reason_code)
            assert result.rationale, pair["test_id"]

    def test_a_clear_majority_produce_a_message(self, outcomes):
        """Restraint is rewarded, silence is not: the engine must do both."""
        sent = [r for _, r in outcomes if r.should_send]

        assert len(sent) >= 15, f"only {len(sent)}/30 produced a message"
        assert len(sent) < 30, "nothing was declined; the no-action path is not exercising"

    def test_all_five_categories_are_represented_in_the_sends(self, outcomes, expanded):
        categories = {
            expanded["merchants"][pair["merchant_id"]]["category_slug"]
            for pair, result in outcomes if result.should_send
        }
        assert categories == {"dentists", "salons", "restaurants", "gyms", "pharmacies"}

    def test_customer_facing_capability_is_exercised(self, outcomes):
        customer_sends = [
            r for _, r in outcomes if r.should_send and r.send_as == "merchant_on_behalf"
        ]
        assert customer_sends, "no customer-facing message was produced"
        for result in customer_sends:
            assert result.customer_id


class TestQuality:
    def test_every_body_passes_the_output_validator(self, outcomes):
        for pair, result in outcomes:
            if result.should_send:
                assert validate_body(result.body, result.plan) == (), pair["test_id"]

    def test_no_body_is_reused(self, outcomes):
        bodies = [r.body for _, r in outcomes if r.should_send]
        assert len(set(bodies)) == len(bodies)

    def test_every_message_contains_a_concrete_number_or_date(self, outcomes):
        """Specificity is the first scored dimension: a message with no
        verifiable figure has nothing for the merchant to check."""
        for pair, result in outcomes:
            if result.should_send:
                assert re.search(r"\d", result.body), pair["test_id"]

    def test_every_message_names_its_recipient(self, outcomes, expanded):
        for pair, result in outcomes:
            if not result.should_send:
                continue
            merchant = expanded["merchants"][pair["merchant_id"]]
            if result.send_as == "merchant_on_behalf":
                customer = expanded["customers"][pair["customer_id"]]
                assert customer["identity"]["name"].split("(")[0].strip() in result.body
            else:
                expected = merchant["identity"].get("owner_first_name", "")
                assert not expected or expected in result.body, pair["test_id"]

    def test_every_send_has_exactly_one_ask(self, outcomes):
        for pair, result in outcomes:
            if result.should_send:
                unquoted = re.sub(r'"[^"]*"', "", result.body)
                assert unquoted.count("?") <= 1, pair["test_id"]

    def test_suppression_keys_are_unique_across_the_set(self, outcomes):
        keys = [r.suppression_key for _, r in outcomes if r.should_send]
        assert len(set(keys)) == len(keys)

    def test_a_research_or_compliance_claim_always_carries_a_citation(self, outcomes):
        """The case studies cap an uncited research claim at 7/10."""
        for pair, result in outcomes:
            if not result.should_send:
                continue
            if result.plan.family in {"knowledge", "compliance"}:
                assert result.plan.citations, pair["test_id"]
                assert result.plan.citations[0] in result.body

    def test_the_cta_taxonomy_is_respected(self, outcomes):
        allowed = {"open_ended", "binary_yes_no", "binary_confirm_cancel",
                   "multi_choice_slot", "none"}
        for _, result in outcomes:
            if result.should_send:
                assert result.cta in allowed

    def test_more_than_one_cta_shape_is_used(self, outcomes):
        shapes = Counter(r.cta for _, r in outcomes if r.should_send)
        assert len(shapes) >= 3, f"CTA variety too low: {dict(shapes)}"


class TestDeterminism:
    def test_the_whole_set_is_reproducible(self, expanded):
        def run() -> list[tuple[str, str, str, str]]:
            out = []
            for pair in expanded["pairs"]:
                merchant = expanded["merchants"].get(pair["merchant_id"])
                trigger = expanded["triggers"].get(pair["trigger_id"])
                customer = expanded["customers"].get(pair.get("customer_id") or "")
                category = expanded["categories"].get((merchant or {}).get("category_slug", ""))
                result = compose(category, merchant, trigger, customer, now=NOW)
                out.append((result.decision, result.body, result.cta, result.suppression_key))
            return out

        assert run() == run()


def test_write_submission_lines(outcomes, tmp_path: Path):
    """The set serialises to the submission JSONL shape (challenge-brief §7.2)."""
    lines = []
    for pair, result in outcomes:
        record = {"test_id": pair["test_id"], **result.to_dict()}
        lines.append(json.dumps(record, ensure_ascii=False))
    target = tmp_path / "submission.jsonl"
    target.write_text("\n".join(lines), encoding="utf-8")

    reloaded = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(reloaded) == 30
    assert all(set(r) == {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
               for r in reloaded)
