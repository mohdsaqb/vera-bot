"""The 30 canonical pairs driven through the endpoints, not through a helper.

`test_canonical_pairs.py` exercises `compose()` directly; this suite pushes each
pair's contexts to `/v1/context` and reads the action back from `/v1/tick`, which
is the only path the judge uses. It also captures the per-pair record the phase
brief asks for — trigger, signal, action, body, CTA, suppression key, rationale,
latency — and asserts the five scored dimensions as properties of the whole set
rather than as expected answers for individual cases.

Skipped when `expanded/` has not been generated:
    python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Any

import pytest

from conftest import NOW, challenge_dir, push_context, tick_now

# Per-call budgets from challenge-testing-brief.md §5 and the summary table.
CONTEXT_BUDGET_MS = 5_000
TICK_BUDGET_MS = 10_000


@lru_cache(maxsize=1)
def _expanded() -> dict[str, Any] | None:
    root = challenge_dir() / "expanded"
    if not (root / "test_pairs.json").is_file():
        return None

    def index(folder: str, key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for path in sorted((root / folder).glob("*.json")):
            payload = json.loads(path.read_text())
            out[payload.get(key, path.stem)] = payload
        return out

    return {
        "pairs": json.loads((root / "test_pairs.json").read_text())["pairs"],
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


@pytest.fixture
def records(client, expanded) -> list[dict[str, Any]]:
    """Run all 30 pairs through the API, one tick each, capturing everything.

    One pair per tick keeps the per-merchant cap from hiding a pair whose
    merchant appears twice in the set.
    """
    out: list[dict[str, Any]] = []
    for pair in expanded["pairs"]:
        merchant = expanded["merchants"][pair["merchant_id"]]
        trigger = expanded["triggers"][pair["trigger_id"]]
        customer = expanded["customers"].get(pair.get("customer_id") or "")
        category = expanded["categories"][merchant["category_slug"]]

        started = time.perf_counter()
        push_context(client, "category", merchant["category_slug"], category)
        push_context(client, "merchant", merchant["merchant_id"], merchant)
        if customer:
            push_context(client, "customer", customer["customer_id"], customer)
        push_context(client, "trigger", trigger["id"], trigger)
        push_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        actions = tick_now(client, [trigger["id"]])
        tick_ms = (time.perf_counter() - started) * 1000

        action = actions[0] if actions else None
        out.append({
            "test_id": pair["test_id"],
            "category": merchant["category_slug"],
            "trigger_id": trigger["id"],
            "trigger_kind": trigger["kind"],
            "scope": trigger["scope"],
            "sent": action is not None,
            "action": action,
            "push_ms": push_ms,
            "tick_ms": tick_ms,
        })
    return out


@pytest.fixture
def sent(records) -> list[dict[str, Any]]:
    return [r for r in records if r["sent"]]


# ---------------------------------------------------------------------------- #
# Coverage through the real path
# ---------------------------------------------------------------------------- #
class TestCoverage:
    def test_every_pair_is_handled(self, records):
        assert len(records) == 30

    def test_a_clear_majority_send(self, sent):
        """Both halves must exercise: restraint is scored, silence is not free."""
        assert len(sent) >= 15, f"only {len(sent)}/30 produced an action"
        assert len(sent) < 30, "nothing was declined; the no-action path is idle"

    def test_all_five_categories_appear(self, sent):
        assert {r["category"] for r in sent} == {
            "dentists", "salons", "restaurants", "gyms", "pharmacies"
        }

    def test_customer_facing_pairs_are_exercised(self, sent):
        customer_sends = [r for r in sent if r["action"]["send_as"] == "merchant_on_behalf"]

        assert customer_sends
        assert all(r["action"]["customer_id"] for r in customer_sends)

    def test_every_action_is_wire_complete(self, sent):
        required = {
            "conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id",
            "template_name", "template_params", "body", "cta", "suppression_key",
            "rationale",
        }
        for record in sent:
            assert set(record["action"]) == required, record["test_id"]
            for field in ("conversation_id", "merchant_id", "send_as", "trigger_id",
                          "body", "cta", "suppression_key", "rationale"):
                assert record["action"][field], f"{record['test_id']}: empty {field}"


# ---------------------------------------------------------------------------- #
# 19. The five scored dimensions, as properties of the set
# ---------------------------------------------------------------------------- #
class TestDecisionQuality:
    def test_the_chosen_trigger_is_the_pair_trigger(self, sent):
        for record in sent:
            assert record["action"]["trigger_id"] == record["trigger_id"]

    def test_every_send_has_a_traceable_reason(self, sent):
        for record in sent:
            rationale = record["action"]["rationale"]
            assert record["trigger_kind"] in rationale, record["test_id"]
            assert "priority" in rationale
            assert "anchored on" in rationale

    def test_declines_are_reasoned_not_silent(self, records, expanded):
        """A pair that sends nothing must have a decision behind it."""
        from engine import compose

        declined = [r for r in records if not r["sent"]]
        assert declined, "the no-action path should be exercised by this set"
        for record in declined:
            trigger = expanded["triggers"][record["trigger_id"]]
            merchant = expanded["merchants"][trigger["merchant_id"]]
            customer = expanded["customers"].get(trigger.get("customer_id") or "")
            result = compose(expanded["categories"][merchant["category_slug"]],
                             merchant, trigger, customer, now=NOW)
            assert result.decision == "no_action"
            assert result.reason_code != "sent"
            assert result.rationale


class TestSpecificity:
    def test_every_message_carries_a_figure(self, sent):
        for record in sent:
            assert re.search(r"\d", record["action"]["body"]), record["test_id"]

    def test_research_and_compliance_claims_are_cited(self, sent):
        for record in sent:
            if record["trigger_kind"] in {"research_digest", "regulation_change",
                                          "supply_alert", "cde_opportunity"}:
                body = record["action"]["body"]
                assert "—" in body and any(
                    marker in body for marker in ("20", "circular", "p.", "calendar", "alert")
                ), f"{record['test_id']} makes a sourced claim without a source"

    def test_no_money_amount_is_unsourced(self, sent, expanded):
        for record in sent:
            merchant = expanded["merchants"][record["action"]["merchant_id"]]
            category = expanded["categories"][merchant["category_slug"]]
            supplied = json.dumps([merchant, category, expanded["triggers"][
                record["trigger_id"]]], ensure_ascii=False)
            for amount in re.findall(r"₹[\d,]+", record["action"]["body"]):
                assert amount in supplied, f"{record['test_id']}: unsourced {amount}"


class TestCategoryFit:
    def test_no_single_template_dominates(self, sent):
        """Openings must differ across categories, not just in the middle."""
        openings = [r["action"]["body"].split("—")[0].strip() for r in sent]

        assert len(set(openings)) >= len({r["category"] for r in sent})

    def test_each_category_uses_more_than_one_action_shape(self, sent):
        by_category: dict[str, set[str]] = {}
        for record in sent:
            by_category.setdefault(record["category"], set()).add(record["action"]["cta"])

        assert sum(len(shapes) for shapes in by_category.values()) > len(by_category)

    def test_dentists_are_addressed_clinically(self, sent):
        dentist_sends = [r for r in sent if r["category"] == "dentists"]

        assert dentist_sends
        for record in dentist_sends:
            body = record["action"]["body"]
            if record["action"]["send_as"] == "vera":
                assert body.startswith("Dr."), record["test_id"]

    def test_no_taboo_vocabulary_anywhere(self, sent, expanded):
        for record in sent:
            merchant = expanded["merchants"][record["action"]["merchant_id"]]
            taboo = expanded["categories"][merchant["category_slug"]]["voice"]["vocab_taboo"]
            lowered = record["action"]["body"].lower()
            for term in taboo:
                phrase = term.split("(")[0].strip().lower()
                assert phrase not in lowered, f"{record['test_id']}: {phrase}"


class TestMerchantFit:
    def test_every_message_names_its_recipient(self, sent, expanded):
        for record in sent:
            action = record["action"]
            if action["send_as"] == "merchant_on_behalf":
                customer = expanded["customers"][action["customer_id"]]
                name = customer["identity"]["name"].split("(")[0].strip()
                assert name in action["body"], record["test_id"]
            else:
                owner = expanded["merchants"][action["merchant_id"]]["identity"].get(
                    "owner_first_name", ""
                )
                assert not owner or owner in action["body"], record["test_id"]

    def test_bodies_are_not_interchangeable(self, sent):
        assert len({r["action"]["body"] for r in sent}) == len(sent)

    def test_customer_messages_carry_no_merchant_analytics(self, sent):
        leaks = ("click-through", "percentage points", "peer", "median", "profile views")
        for record in sent:
            if record["action"]["send_as"] != "merchant_on_behalf":
                continue
            lowered = record["action"]["body"].lower()
            for term in leaks:
                assert term not in lowered, f"{record['test_id']} leaked {term}"


class TestEngagement:
    def test_exactly_one_ask(self, sent):
        for record in sent:
            unquoted = re.sub(r'"[^"]*"', "", record["action"]["body"])
            assert unquoted.count("?") <= 1, record["test_id"]

    def test_the_cta_is_from_the_taxonomy(self, sent):
        allowed = {"open_ended", "binary_yes_no", "binary_confirm_cancel",
                   "multi_choice_slot", "none"}
        for record in sent:
            assert record["action"]["cta"] in allowed

    def test_more_than_one_cta_shape_is_used(self, sent):
        shapes = Counter(r["action"]["cta"] for r in sent)

        assert len(shapes) >= 3, f"CTA variety too low: {dict(shapes)}"

    def test_the_ask_is_answerable_in_one_move(self, sent):
        for record in sent:
            body = record["action"]["body"].lower()
            assert any(
                marker in body
                for marker in ("?", "reply", "confirm", "kijiye", "dijiye", "dun")
            ), record["test_id"]

    def test_bodies_stay_readable(self, sent):
        for record in sent:
            assert 80 <= len(record["action"]["body"]) <= 600, (
                f"{record['test_id']}: {len(record['action']['body'])} chars"
            )


# ---------------------------------------------------------------------------- #
# 22. Latency
# ---------------------------------------------------------------------------- #
class TestLatency:
    def test_context_pushes_are_well_inside_budget(self, records):
        worst = max(r["push_ms"] for r in records)

        assert worst < CONTEXT_BUDGET_MS, f"{worst:.0f}ms for four pushes"

    def test_ticks_are_well_inside_budget(self, records):
        worst = max(r["tick_ms"] for r in records)

        assert worst < TICK_BUDGET_MS, f"{worst:.0f}ms"

    def test_health_and_metadata_are_trivial(self, client):
        started = time.perf_counter()
        for _ in range(20):
            client.get("/v1/healthz")
            client.get("/v1/metadata")
        per_call_ms = (time.perf_counter() - started) * 1000 / 40

        assert per_call_ms < 50, f"{per_call_ms:.1f}ms per call"


# ---------------------------------------------------------------------------- #
# Determinism of the whole set
# ---------------------------------------------------------------------------- #
def test_the_whole_set_is_reproducible(client, expanded):
    from state import reset_state

    def run() -> list[tuple[str, str, str]]:
        reset_state()
        out = []
        for pair in expanded["pairs"]:
            merchant = expanded["merchants"][pair["merchant_id"]]
            trigger = expanded["triggers"][pair["trigger_id"]]
            customer = expanded["customers"].get(pair.get("customer_id") or "")
            push_context(client, "category", merchant["category_slug"],
                         expanded["categories"][merchant["category_slug"]])
            push_context(client, "merchant", merchant["merchant_id"], merchant)
            if customer:
                push_context(client, "customer", customer["customer_id"], customer)
            push_context(client, "trigger", trigger["id"], trigger)
            actions = tick_now(client, [trigger["id"]])
            out.append((
                pair["test_id"],
                actions[0]["body"] if actions else "",
                actions[0]["suppression_key"] if actions else "",
            ))
        return out

    assert run() == run()
