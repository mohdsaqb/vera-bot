"""Bridge between the stored contexts and the decision engine.

`/v1/tick` hands over a list of trigger ids and a simulated time. This module
resolves each id into the four contexts the engine needs, ranks the eligible
triggers, applies the per-tick sending discipline and returns actions in the
judge's shape.

Sending discipline (challenge-testing-brief.md §5 and §10, and the FAQ):
  * at most one action per merchant per tick — several stories may qualify, but
    the merchant gets the strongest one and the rest wait for a later tick;
  * at most `MAX_ACTIONS_PER_TICK` actions in total;
  * nothing already sent under the same suppression key, unless the context it
    was built from has been refreshed *and* the message has changed;
  * no body already sent in that conversation;
  * nothing to a merchant who has opted out.

Every decision reads the contexts out of the store at the moment it is made, so a
version pushed between two ticks is used by the second one. Nothing is cached
here — the store is the only copy.

Wording is delegated to the configured writer under a per-tick budget: `/v1/tick`
has a 10-second latency budget, so the first few messages are generated and the
rest render deterministically rather than risking the whole tick.

Every action opens (or continues) a conversation, so the reply that comes back
lands on a thread that knows what it was about.

No decision logic lives here — that is `engine/`; this is resolution, ordering
and bookkeeping.
"""

from __future__ import annotations

import logging
from typing import Any

from engine import compose
from engine.normalize import (
    normalize_category,
    normalize_customer,
    normalize_merchant,
    normalize_trigger,
    parse_iso,
)
from engine.triggers import evaluate_trigger, rank_triggers
from engine.types import ComposedMessage, MessageWriter, NormalizedTrigger, TriggerEvaluation
from services.context_store import ContextStore
from services.conversation_store import ConversationStore
from services.llm_service import TickBudget
from services.suppression import CONTEXT_REFRESHED, SuppressionLedger

logger = logging.getLogger("vera.decisions")

# The judge caps a tick at 20 actions; staying under it is the bot's job.
MAX_ACTIONS_PER_TICK = 20

# How far past every available expiry the tick's clock has to sit before it is
# read as being on a different timeline rather than as events having lapsed.
# A month is comfortably longer than any event decay in the dataset (the shortest
# trigger window is a single evening) and comfortably shorter than the gap that
# appears when a wall clock is used against a dated dataset.
OFF_TIMELINE_DAYS = 30


class DecisionService:
    """Turns a tick into a list of proactive actions."""

    def __init__(
        self,
        store: ContextStore,
        ledger: SuppressionLedger,
        conversations: ConversationStore | None = None,
        writer: MessageWriter | None = None,
    ) -> None:
        self._store = store
        self._ledger = ledger
        self._conversations = conversations
        self._writer = writer

    # ------------------------------------------------------------------ reads
    def _contexts_for(
        self, trigger: NormalizedTrigger
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Resolve (category, merchant, customer) payloads for one trigger."""
        merchant = self._store.get_payload("merchant", trigger.merchant_id)
        category = None
        if merchant:
            slug = merchant.get("category_slug")
            if isinstance(slug, str) and slug:
                category = self._store.get_payload("category", slug)
        customer = (
            self._store.get_payload("customer", trigger.customer_id)
            if trigger.customer_id
            else None
        )
        return category, merchant, customer

    def _context_version(self, trigger: NormalizedTrigger) -> int:
        """A monotonic stamp for the contexts behind one trigger.

        Summing the held versions gives a value that rises whenever the judge
        pushes a newer version of anything this decision depends on, which is all
        the suppression rule needs to tell "refreshed" from "unchanged".
        """
        merchant_id = trigger.merchant_id
        slug = ""
        merchant = self._store.get_payload("merchant", merchant_id)
        if merchant:
            candidate = merchant.get("category_slug")
            slug = candidate if isinstance(candidate, str) else ""
        versions = (
            self._store.get_version("trigger", trigger.id) or 0,
            self._store.get_version("merchant", merchant_id) or 0,
            self._store.get_version("category", slug) or 0 if slug else 0,
            self._store.get_version("customer", trigger.customer_id) or 0
            if trigger.customer_id
            else 0,
        )
        return sum(versions)

    def _reference_time(
        self, triggers: list[NormalizedTrigger], now: str | None
    ) -> str | None:
        """The time to judge this tick against, or None when it cannot be located.

        Expiry only means something relative to the timeline the triggers live on.
        The dataset's events run from April to December 2026, and the tick's `now`
        is normally on that timeline — but `judge_simulator.py` sends the machine's
        wall clock, which can sit outside it entirely. Taking that literally would
        expire every trigger at once and answer every tick with silence.

        Two conditions have to hold before that is treated as a wrong clock rather
        than as genuinely lapsed events, because the difference matters and a
        single data point cannot tell them apart:

          * more than one trigger carries an expiry — one expired trigger is an
            expired trigger, not evidence about a calendar;
          * every one of them lapsed more than `OFF_TIMELINE_DAYS` ago — events
            decaying over days is ordinary, a whole queue lapsing months ago is a
            different timeline.

        When both hold, time-based judgements are withheld for the tick rather than
        guessed at: the engine already handles a missing reference time by skipping
        expiry and elapsed-day reasoning and using neutral time pressure. Every
        other tick keeps strict per-trigger expiry, which is the case that matters —
        an expired trigger inside a live set is still dropped.
        """
        if now is None:
            return None
        reference = parse_iso(now)
        if reference is None:
            return None

        horizons = [
            expiry for trigger in triggers
            if (expiry := parse_iso(trigger.expires_at)) is not None
        ]
        if len(horizons) < 2 or reference <= max(horizons):
            return now
        if (reference - max(horizons)).days < OFF_TIMELINE_DAYS:
            return now

        logger.warning(
            "tick: now=%s sits %d days past the newest of %d trigger expiries (%s); "
            "treating the reference clock as off-timeline and withholding "
            "time-based judgements for this tick",
            now, (reference - max(horizons)).days, len(horizons), max(horizons).date(),
        )
        return None

    def evaluate_available(
        self, trigger_ids: list[str], now: str | None
    ) -> list[tuple[TriggerEvaluation, NormalizedTrigger]]:
        """Evaluate every available trigger, best-first.

        Unknown ids are skipped rather than failing the tick — the judge may name
        a trigger whose context has not been pushed yet.
        """
        evaluations: list[TriggerEvaluation] = []
        triggers: dict[str, NormalizedTrigger] = {}
        resolved: list[NormalizedTrigger] = []

        for trigger_id in trigger_ids:
            payload = self._store.get_payload("trigger", trigger_id)
            trigger = normalize_trigger(payload)
            if trigger is None:
                logger.debug("tick: no stored context for trigger %s", trigger_id)
                continue
            resolved.append(trigger)
            # The stored id is authoritative for lookups even if the payload's own
            # `id` field differs from the context_id it was pushed under.
            triggers[trigger.id or trigger_id] = trigger

        reference = self._reference_time(resolved, now)
        for trigger in triggers.values():
            category, merchant, customer = self._contexts_for(trigger)
            evaluations.append(
                evaluate_trigger(
                    normalize_category(category),
                    normalize_merchant(merchant),
                    trigger,
                    normalize_customer(customer),
                    reference,
                )
            )

        ranked = rank_triggers(evaluations, triggers)
        return [(evaluation, triggers[evaluation.trigger_id]) for evaluation in ranked]

    # ----------------------------------------------------------------- decide
    def decide(self, trigger_ids: list[str], now: str | None) -> list[ComposedMessage]:
        """Compose the messages to send this tick, in priority order."""
        results: list[ComposedMessage] = []
        used_merchants: set[str] = set()
        writer = self._budgeted_writer()
        ranked = self.evaluate_available(trigger_ids, now)
        reference = self._reference_time([trigger for _, trigger in ranked], now)

        for _, trigger in ranked:
            if len(results) >= MAX_ACTIONS_PER_TICK:
                break
            if trigger.merchant_id in used_merchants:
                continue
            if self._ledger.is_suppressed(f"optout:{trigger.merchant_id}"):
                logger.debug("tick: %s has opted out; skipping", trigger.merchant_id)
                continue
            if self._conversations is not None and self._conversations.is_merchant_closed(
                trigger.merchant_id
            ):
                logger.debug("tick: %s closed the conversation; skipping", trigger.merchant_id)
                continue

            category, merchant, customer = self._contexts_for(trigger)
            composed = compose(
                category, merchant, trigger.raw, customer, now=reference, writer=writer,
            )
            if not composed.should_send:
                continue

            context_version = self._context_version(trigger)
            allowed, reason = self._ledger.should_send(
                composed.suppression_key, composed.body, context_version
            )
            if not allowed:
                logger.debug(
                    "tick: holding %s (%s, key=%s)",
                    trigger.id, reason, composed.suppression_key,
                )
                continue
            if self._ledger.is_repeat(composed.conversation_id, composed.body):
                logger.debug("tick: skipped repeat body in %s", composed.conversation_id)
                continue
            if reason == CONTEXT_REFRESHED:
                logger.info(
                    "tick: resending %s — context refreshed to v%d",
                    composed.suppression_key, context_version,
                )

            self._ledger.record(
                composed.suppression_key,
                composed.conversation_id,
                composed.body,
                context_version,
            )
            self._open_conversation(composed, now)
            used_merchants.add(trigger.merchant_id)
            results.append(composed)

        return results

    def _budgeted_writer(self) -> MessageWriter | None:
        """Bind this tick's generation budget to the writer, when it takes one.

        Duck-typed so a test double needs nothing beyond `write(brief)`.
        """
        writer = self._writer
        if writer is None:
            return None
        binder = getattr(writer, "with_budget", None)
        return binder(TickBudget.start()) if callable(binder) else writer

    def _open_conversation(self, composed: ComposedMessage, now: str | None) -> None:
        """Record the outbound message so `/v1/reply` has context for it."""
        if self._conversations is None or composed.plan is None:
            return
        plan = composed.plan
        self._conversations.open_from_action(
            composed.conversation_id,
            merchant_id=composed.merchant_id,
            customer_id=composed.customer_id,
            trigger_id=composed.trigger_id,
            family=plan.family,
            action_key=plan.action.key,
            cta=composed.cta,
            send_as=composed.send_as,
            audience=plan.audience,
            language=plan.language,
            category_slug=plan.category_slug,
            suppression_key=composed.suppression_key,
            brief=composed.brief,
            body=composed.body,
            at=now or "",
        )

    def actions(self, trigger_ids: list[str], now: str | None) -> list[dict[str, Any]]:
        """The decisions for this tick, in the judge's action shape."""
        return [composed.to_action() for composed in self.decide(trigger_ids, now)]
