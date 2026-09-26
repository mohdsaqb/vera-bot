"""Pytest fixtures shared by the test suite.

Lives at the project root so `app`, `models`, `state` and `services` import the
same way in tests as they do under uvicorn.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import app
from services.context_store import ContextStore
from services.suppression import SuppressionLedger
from state import get_context_store, get_suppression_ledger, reset_state

# Variables that change how the code behaves rather than what it says. A developer
# with a real `.env` — which anyone who has deployed will have — would otherwise be
# running a different suite from CI: `build_writer()` would return an LLMWriter, and
# the reasoning defaults would already be populated. Tests that want a provider
# configure one explicitly by constructing `Settings(...)`.
BEHAVIOURAL_ENV_VARS = (
    "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_TOKENS", "LLM_STRUCTURED_METHOD", "LLM_REASONING_EFFORT",
    "LLM_REASONING_FORMAT", "LLM_MAX_CALLS_PER_TICK", "LLM_TICK_BUDGET_SECONDS",
    "GROQ_API_KEY",
)


@pytest.fixture(autouse=True)
def neutral_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test against the code's own defaults, not the developer's `.env`.

    Cleared before and after, with the settings cache invalidated either side so a
    value read during one test cannot persist into the next.
    """
    from config import get_settings

    for name in BEHAVIOURAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_state() -> Iterator[None]:
    """Reset all process-wide state around every test.

    The context store and the suppression ledger are singletons by design (the
    judge pushes context across many calls and the bot must remember what it
    already sent), so tests must not inherit each other's state.
    """
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def store() -> ContextStore:
    """The store the app is using, already emptied by `clean_state`."""
    return get_context_store()


@pytest.fixture
def fresh_store() -> ContextStore:
    """A standalone store, for unit tests that should not touch app state."""
    return ContextStore()


def context_push(
    scope: str = "merchant",
    context_id: str = "m_001_drmeera_dentist_delhi",
    version: int = 1,
    payload: dict | None = None,
    delivered_at: str | None = "2026-04-26T09:45:30Z",
) -> dict:
    """Build a valid `POST /v1/context` body."""
    body = {
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload if payload is not None else {"merchant_id": context_id},
    }
    if delivered_at is not None:
        body["delivered_at"] = delivered_at
    return body


# --------------------------------------------------------------------------- #
# Challenge dataset
# --------------------------------------------------------------------------- #
# Tests read the real seed files rather than hand-written fixtures, so they fail
# if the engine stops matching the actual challenge schema. Seeds are used (not
# the generated `expanded/` tree) because they are committed and always present.
CHALLENGE_ENV_VAR = "CHALLENGE_DIR"
DEFAULT_CHALLENGE_DIR = Path(__file__).resolve().parent.parent / "magicpin-ai-challenge"

# A reference time inside the seed dataset's own window (the briefs are dated
# 2026-04-26), so expiry and elapsed-day logic has something real to measure.
NOW = "2026-04-26T10:35:00Z"


def challenge_dir() -> Path:
    return Path(os.environ.get(CHALLENGE_ENV_VAR, DEFAULT_CHALLENGE_DIR))


@dataclass(frozen=True)
class Dataset:
    """The seed dataset, indexed by id."""

    categories: dict[str, dict[str, Any]]
    merchants: dict[str, dict[str, Any]]
    customers: dict[str, dict[str, Any]]
    triggers: dict[str, dict[str, Any]]

    def category_for(self, merchant_id: str) -> dict[str, Any]:
        return self.categories[self.merchants[merchant_id]["category_slug"]]

    def compose_args(self, trigger_id: str) -> tuple[dict, dict, dict, dict | None]:
        """(category, merchant, trigger, customer) for one seed trigger."""
        trigger = self.triggers[trigger_id]
        merchant = self.merchants[trigger["merchant_id"]]
        customer = self.customers.get(trigger.get("customer_id") or "")
        return self.category_for(merchant["merchant_id"]), merchant, trigger, customer


@lru_cache(maxsize=1)
def _load_dataset() -> Dataset | None:
    root = challenge_dir() / "dataset"
    if not root.is_dir():
        return None
    categories = {
        data["slug"]: data
        for path in sorted((root / "categories").glob("*.json"))
        if (data := json.loads(path.read_text()))
    }
    def _indexed(filename: str, container: str, key: str) -> dict[str, dict[str, Any]]:
        payload = json.loads((root / filename).read_text())
        return {item[key]: item for item in payload[container] if key in item}

    return Dataset(
        categories=categories,
        merchants=_indexed("merchants_seed.json", "merchants", "merchant_id"),
        customers=_indexed("customers_seed.json", "customers", "customer_id"),
        triggers=_indexed("triggers_seed.json", "triggers", "id"),
    )


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """The seed dataset. Skips the test when the challenge files are not present."""
    loaded = _load_dataset()
    if loaded is None:
        pytest.skip(f"challenge dataset not found at {challenge_dir()}")
    return loaded


@pytest.fixture
def now() -> str:
    return NOW


@pytest.fixture
def ledger() -> SuppressionLedger:
    """The process-wide suppression ledger, already emptied by `clean_state`."""
    return get_suppression_ledger()


def variant(_context: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Deep copy of a context payload with top-level keys replaced.

    The positional name is underscored so `payload=` can itself be overridden —
    trigger contexts have a field by that name.
    """
    clone = copy.deepcopy(_context)
    clone.update(copy.deepcopy(overrides))
    return clone


def deep_set(_context: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Deep copy of a payload with one nested field replaced.

    `deep_set(merchant, "performance.delta_7d.calls_pct", -0.4)`.
    """
    clone = copy.deepcopy(_context)
    target = clone
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    return clone


# --------------------------------------------------------------------------- #
# Writers (no test makes a real model call)
# --------------------------------------------------------------------------- #
@dataclass
class FakeWriter:
    """A stand-in for the wording layer.

    `bodies` are returned in order, one per call; `raises` makes the call blow up
    so the fallback path can be exercised. Records every brief it was handed, so
    a test can assert what the writer was — and was not — allowed to see.
    """

    bodies: list[str] = field(default_factory=list)
    raises: Exception | None = None
    source: str = "llm"
    name: str = "llm"
    seen: list[Any] = field(default_factory=list)
    calls: int = 0

    def write(self, brief, budget=None):  # noqa: ANN001, ANN202, ARG002 - writer interface
        from engine.types import GenerationResult

        self.calls += 1
        self.seen.append(brief)
        if self.raises is not None:
            raise self.raises
        if not self.bodies:
            return GenerationResult(body=brief.deterministic_body, source="deterministic")
        body = self.bodies[min(self.calls - 1, len(self.bodies) - 1)]
        return GenerationResult(body=body, source=self.source, model="fake-model")

    def unavailable_reason(self) -> str:
        return ""


@pytest.fixture
def fake_writer() -> FakeWriter:
    return FakeWriter()


@pytest.fixture
def brief(dataset) -> Any:
    """A real `GenerationBrief`, built by the engine from seed data.

    Using the engine's own projection rather than a hand-made object keeps these
    tests honest about what a writer actually receives.
    """
    from engine.brief import brief_for_plan
    from engine.compose import build_plan
    from engine.normalize import (
        normalize_category,
        normalize_merchant,
        normalize_trigger,
    )
    from engine.triggers import evaluate_trigger

    category_payload, merchant_payload, trigger_payload, _ = dataset.compose_args(
        "trg_004_perf_dip_bharat"
    )
    category = normalize_category(category_payload)
    merchant = normalize_merchant(merchant_payload)
    trigger = normalize_trigger(trigger_payload)
    evaluation = evaluate_trigger(category, merchant, trigger, None, NOW)
    plan = build_plan(category, merchant, trigger, None, evaluation, NOW)
    return brief_for_plan(plan, category, merchant, None)


# --------------------------------------------------------------------------- #
# HTTP drivers
# --------------------------------------------------------------------------- #
# Integration tests go through the endpoints rather than calling helpers, because
# the endpoints are all the judge ever sees. These four wrappers are the judge's
# own call shapes, kept here so every suite drives the API identically.
def push_context(
    client, scope: str, context_id: str, payload: dict, version: int = 1
):
    """`POST /v1/context`. Returns the raw response so status can be asserted."""
    return client.post(
        "/v1/context",
        json={"scope": scope, "context_id": context_id, "version": version,
              "payload": payload, "delivered_at": NOW},
    )


def tick_now(client, trigger_ids: list[str], now: str = NOW) -> list[dict]:
    """`POST /v1/tick`. Returns the actions, asserting the call itself succeeded."""
    response = client.post("/v1/tick", json={"now": now, "available_triggers": trigger_ids})
    assert response.status_code == 200, response.text
    return response.json()["actions"]


def reply_to(
    client, conversation_id: str, message: str, merchant_id: str = "m_001", **extra
) -> dict:
    """`POST /v1/reply`. Returns the decoded answer."""
    body = {
        "conversation_id": conversation_id, "merchant_id": merchant_id,
        "from_role": "merchant", "message": message, "received_at": NOW,
        "turn_number": 2, **extra,
    }
    response = client.post("/v1/reply", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def load_contexts(client, dataset: Dataset, trigger_ids: list[str]) -> None:
    """Push everything a set of triggers depends on, as the judge's warmup does."""
    for trigger_id in trigger_ids:
        trigger = dataset.triggers[trigger_id]
        merchant = dataset.merchants[trigger["merchant_id"]]
        push_context(client, "category", merchant["category_slug"],
                     dataset.categories[merchant["category_slug"]])
        push_context(client, "merchant", merchant["merchant_id"], merchant)
        if trigger.get("customer_id"):
            push_context(client, "customer", trigger["customer_id"],
                         dataset.customers[trigger["customer_id"]])
        push_context(client, "trigger", trigger_id, trigger)
