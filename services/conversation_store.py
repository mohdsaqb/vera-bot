"""Conversation state for messages in flight.

The judge opens a conversation by delivering one of the bot's `/v1/tick` actions
and then posts replies to `/v1/reply`, so the bot has to remember what it said,
what came back, and where the thread stands. In-memory and process-level, like
the context store: the test window is one process and the privacy rule forbids
keeping any of it afterwards.

`ConversationState` itself lives in `engine/types.py` — it is a domain type the
decision layer reasons about, and keeping it there stops `engine/` having to
import a service.

Two counters deserve explanation:

* `consecutive_auto_replies` — how many machine replies have arrived in a row on
  this thread, which is what decides between nudging, backing off and closing.
* per-merchant auto-reply totals — the same phone answers every thread, so a
  merchant whose WhatsApp Business auto-reply is on will produce canned text
  across several conversation ids. `judge_simulator.py` does exactly that, using
  a fresh conversation id for each of its four canned turns, so counting only
  per-thread would never notice.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from engine.policy import MAX_UNANSWERED_NUDGES
from engine.types import ConversationState, GenerationBrief, Turn

__all__ = [
    "MAX_UNANSWERED_NUDGES",
    "ConversationState",
    "ConversationStore",
    "Turn",
]


class ConversationStore:
    """Thread-safe store of conversations, plus per-merchant auto-reply counts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conversations: dict[str, ConversationState] = {}
        self._merchant_auto_replies: dict[str, int] = {}

    # ---------------------------------------------------------------- lookup
    def get(self, conversation_id: str) -> ConversationState | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def get_or_create(
        self, conversation_id: str, **fields: Any
    ) -> ConversationState:
        """Return the thread, opening it if the judge got there first.

        `/v1/reply` can name a conversation the bot never opened — the replay
        scenarios do — so an unknown id is a new thread rather than an error.
        """
        with self._lock:
            existing = self._conversations.get(conversation_id)
            if existing is not None:
                return existing
            state = ConversationState(conversation_id=conversation_id, **fields)
            self._conversations[conversation_id] = state
            return state

    def save(self, state: ConversationState) -> ConversationState:
        with self._lock:
            self._conversations[state.conversation_id] = state
            return state

    # --------------------------------------------------------------- opening
    def open_from_action(
        self,
        conversation_id: str,
        *,
        merchant_id: str,
        customer_id: str | None,
        trigger_id: str,
        family: str,
        action_key: str,
        cta: str,
        send_as: str,
        audience: str,
        language: str,
        category_slug: str,
        suppression_key: str,
        brief: GenerationBrief | None,
        body: str,
        at: str = "",
    ) -> ConversationState:
        """Record an outbound message, opening the thread if it is new.

        Called for every action `/v1/tick` returns, so the reply that comes back
        lands on a thread that knows what it is about.
        """
        with self._lock:
            state = self._conversations.get(conversation_id) or ConversationState(
                conversation_id=conversation_id
            )
            state = replace(
                state,
                merchant_id=merchant_id or state.merchant_id,
                customer_id=customer_id if customer_id is not None else state.customer_id,
                trigger_id=trigger_id or state.trigger_id,
                family=family or state.family,
                action_key=action_key or state.action_key,
                cta=cta or state.cta,
                send_as=send_as or state.send_as,
                audience=audience or state.audience,
                language=language or state.language,
                category_slug=category_slug or state.category_slug,
                suppression_key=suppression_key or state.suppression_key,
                brief=brief if brief is not None else state.brief,
                turns=(*state.turns, Turn("outbound", body, at=at, cta=cta)),
                turn_count=state.turn_count + 1,
                unanswered_nudges=state.unanswered_nudges + 1,
                active=True,
            )
            self._conversations[conversation_id] = state
            return state

    # ------------------------------------------------------------ transitions
    def record_inbound(
        self, state: ConversationState, body: str, intent: str, at: str = ""
    ) -> ConversationState:
        """Record an inbound turn and update the counters that depend on it."""
        is_auto = intent == "auto_reply"
        with self._lock:
            if is_auto and state.merchant_id:
                self._merchant_auto_replies[state.merchant_id] = (
                    self._merchant_auto_replies.get(state.merchant_id, 0) + 1
                )
            updated = replace(
                state,
                turns=(*state.turns, Turn("inbound", body, at=at, intent=intent)),
                turn_count=state.turn_count + 1,
                last_intent=intent,
                last_inbound_at=at or state.last_inbound_at,
                consecutive_auto_replies=(
                    state.consecutive_auto_replies + 1 if is_auto else 0
                ),
                unanswered_nudges=0 if not is_auto else state.unanswered_nudges,
            )
            self._conversations[updated.conversation_id] = updated
            return updated

    def record_outbound(
        self, state: ConversationState, body: str, cta: str, at: str = ""
    ) -> ConversationState:
        with self._lock:
            updated = replace(
                state,
                turns=(*state.turns, Turn("outbound", body, at=at, cta=cta)),
                turn_count=state.turn_count + 1,
                unanswered_nudges=state.unanswered_nudges + 1,
            )
            self._conversations[updated.conversation_id] = updated
            return updated

    def close(self, state: ConversationState, reason: str) -> ConversationState:
        with self._lock:
            updated = replace(
                state, active=False, should_end=True, end_reason=reason
            )
            self._conversations[updated.conversation_id] = updated
            return updated

    def mark_delay(self, state: ConversationState, seconds: int) -> ConversationState:
        with self._lock:
            updated = replace(state, delay_requested_seconds=seconds)
            self._conversations[updated.conversation_id] = updated
            return updated

    # ------------------------------------------------------- merchant counters
    def merchant_auto_replies(self, merchant_id: str) -> int:
        """How many canned replies this merchant's number has sent, all threads."""
        with self._lock:
            return self._merchant_auto_replies.get(merchant_id, 0)

    def is_merchant_closed(self, merchant_id: str) -> bool:
        """True when a thread with this merchant was closed on their say-so.

        An opt-out or a hostile exit applies to the merchant, not just the thread
        it happened on, so later triggers must not reopen the conversation.
        """
        if not merchant_id:
            return False
        with self._lock:
            return any(
                state.merchant_id == merchant_id
                and state.should_end
                and state.end_reason in {"opted_out", "hostile"}
                for state in self._conversations.values()
            )

    # ------------------------------------------------------------------ reset
    def clear(self) -> int:
        with self._lock:
            count = len(self._conversations)
            self._conversations.clear()
            self._merchant_auto_replies.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._conversations)
