"""`/v1/reply` orchestration.

Order of work, and which layer owns each step:

    1. find or open the thread                  ConversationStore
    2. classify the inbound message              engine.intent      (deterministic)
    3. decide send / wait / end                   engine.reply       (deterministic)
    4. reword the draft, if a writer is set       services.llm_service
    5. validate the wording                       services.message_validator
    6. record the turn and answer                 ConversationStore

Steps 2 and 3 are the decision and never involve a model. Step 4 may only change
how the draft reads; if it changes anything else, step 5 catches it and the draft
is sent instead.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from engine.intent import ACCEPT, ACTION_REQUEST, classify_reply
from engine.reply import ReplyDecision, brief_for_reply, decide_reply, re_qualifies
from engine.types import ConversationState, MessageWriter, ReplyOutcome
from services.conversation_store import ConversationStore
from services.llm_service import DeterministicWriter
from services.message_validator import validate_generated_body
from services.suppression import SuppressionLedger

logger = logging.getLogger("vera.reply")

# End reasons that silence the merchant rather than the thread.
# api-call-examples.md §2.6 suppresses "this conversation_id" for a plain refusal;
# §4.3 suppresses "all triggers for this merchant" for a hostile exit. Treating a
# refusal as the second kind would stop every other story — including
# customer-facing ones — over one declined offer.
MERCHANT_WIDE_END_REASONS = frozenset({"opted_out", "hostile"})


class ReplyService:
    """Answers one inbound turn."""

    def __init__(
        self,
        conversations: ConversationStore,
        ledger: SuppressionLedger,
        writer: MessageWriter | None = None,
    ) -> None:
        self._conversations = conversations
        self._ledger = ledger
        self._writer = writer or DeterministicWriter()

    def handle(
        self,
        conversation_id: str,
        message: str,
        *,
        merchant_id: str = "",
        customer_id: str | None = None,
        from_role: str = "merchant",
        received_at: str = "",
    ) -> ReplyOutcome:
        """Decide and compose the answer to one inbound message."""
        state = self._conversations.get_or_create(
            conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            audience="customer" if from_role == "customer" else "merchant",
        )
        if merchant_id and not state.merchant_id:
            state = self._conversations.save(replace(state, merchant_id=merchant_id))

        result = classify_reply(
            message,
            previous_messages=state.inbound_bodies,
            expects_slot=state.expects_slot,
        )
        state = self._conversations.record_inbound(
            state, message, result.intent, at=received_at
        )

        decision = decide_reply(
            state,
            message,
            result,
            merchant_auto_replies=self._conversations.merchant_auto_replies(
                state.merchant_id
            ),
        )

        if decision.action == "end":
            self._conversations.close(state, decision.end_reason or "ended")
            if decision.end_reason in MERCHANT_WIDE_END_REASONS:
                # An opt-out or a hostile exit is about the channel, so it applies
                # to the merchant. A plain "not interested" is about the story: the
                # contract suppresses that conversation, and the story's own
                # suppression key already prevents it being told again.
                self._ledger.record(
                    f"optout:{state.merchant_id}", state.conversation_id, ""
                )
            logger.info(
                "reply %s: intent=%s -> end (%s)",
                conversation_id, result.intent, decision.end_reason,
            )
            return ReplyOutcome(
                action="end", intent=result.intent, rationale=decision.rationale,
            )

        if decision.action == "wait":
            self._conversations.mark_delay(state, decision.wait_seconds)
            logger.info(
                "reply %s: intent=%s -> wait %ss",
                conversation_id, result.intent, decision.wait_seconds,
            )
            return ReplyOutcome(
                action="wait",
                intent=result.intent,
                rationale=decision.rationale,
                wait_seconds=decision.wait_seconds,
            )

        body, source = self._word(state, decision, message)

        # A body already sent on this thread would earn the anti-repetition
        # penalty, so the thread backs off instead of repeating itself.
        if state.already_sent(body):
            logger.info("reply %s: draft repeats an earlier turn; waiting", conversation_id)
            return ReplyOutcome(
                action="wait",
                intent=result.intent,
                rationale=(
                    "The only reply available repeats what was already sent on this "
                    "thread; backing off instead of repeating it"
                ),
                wait_seconds=decision.wait_seconds or 3_600,
            )

        self._conversations.record_outbound(state, body, decision.cta, at=received_at)
        logger.info(
            "reply %s: intent=%s -> send (%s wording)",
            conversation_id, result.intent, source,
        )
        return ReplyOutcome(
            action="send",
            intent=result.intent,
            rationale=decision.rationale,
            body=body,
            cta=decision.cta,
            body_source=source,
        )

    # ----------------------------------------------------------------- wording
    def _word(
        self, state: ConversationState, decision: ReplyDecision, message: str
    ) -> tuple[str, str]:
        """Reword the draft if a writer is configured, else keep the draft.

        Falls back on any validation problem, and additionally re-checks the one
        thing a reworded acceptance must never do: ask another qualifying
        question.
        """
        brief = brief_for_reply(state, decision, message)
        try:
            generated = self._writer.write(brief)
        except Exception as exc:  # pragma: no cover - writers already swallow
            logger.warning("writer raised %s; using the draft", type(exc).__name__)
            return decision.body, "deterministic"

        if not generated.used_llm or not generated.body:
            return decision.body, "deterministic"

        problems = validate_generated_body(generated.body, brief)
        if problems:
            logger.info("reply wording rejected (%s)", ", ".join(problems[:3]))
            return decision.body, "deterministic"
        if decision.intent in {ACCEPT, ACTION_REQUEST} and re_qualifies(generated.body):
            logger.info("reply wording re-qualified after a commitment; using the draft")
            return decision.body, "deterministic"
        return generated.body.strip(), "llm"
