"""Process-wide runtime state.

The judge pushes context over many HTTP calls, expects the bot to remember all of
it for the whole test window, and expects conversations to survive between a
`/v1/tick` send and the `/v1/reply` that answers it. So each store is a single
process-level singleton rather than per-request state, reached through a getter
so an implementation can be swapped (or a test can reset it) without touching
handler code.

Four things are held: the pushed contexts, what has already been sent, the
conversations in flight, and the writer that words messages. `reset_state()`
clears all of them.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from config import get_settings
from services.context_store import ContextStore
from services.conversation_store import ConversationStore
from services.decision_service import DecisionService
from services.llm_service import build_writer
from services.reply_service import ReplyService
from services.suppression import SuppressionLedger

# Monotonic clock for uptime (immune to wall-clock adjustments) plus the
# wall-clock start time, used as the /v1/metadata fallback timestamp.
_STARTED_MONOTONIC = time.monotonic()
STARTED_AT = datetime.now(timezone.utc)

_context_store = ContextStore()
_suppression_ledger = SuppressionLedger()
_conversation_store = ConversationStore()

# One writer for the process: building it may create an HTTP client, and with no
# provider configured this is the deterministic writer and costs nothing.
_writer = build_writer()


def get_context_store() -> ContextStore:
    """Return the process-wide context store."""
    return _context_store


def get_suppression_ledger() -> SuppressionLedger:
    """Return the process-wide record of what has already been sent."""
    return _suppression_ledger


def get_conversation_store() -> ConversationStore:
    """Return the process-wide store of conversations in flight."""
    return _conversation_store


def get_decision_service() -> DecisionService:
    """Return a decision service bound to the process-wide state.

    Cheap to build and stateless in itself, so a fresh instance per call keeps
    the wiring visible without another singleton to reset.
    """
    return DecisionService(
        _context_store, _suppression_ledger, _conversation_store, _writer
    )


def get_reply_service() -> ReplyService:
    """Return a reply service bound to the process-wide state."""
    return ReplyService(_conversation_store, _suppression_ledger, _writer)


def llm_status() -> dict[str, object]:
    """Whether the wording layer is live, for logs and `/v1/healthz`.

    Reports the configured model name but never the key — there is no code path
    from the key to a response body.
    """
    settings = get_settings()
    is_llm_writer = getattr(_writer, "name", "") == "llm"
    reason = getattr(
        _writer, "unavailable_reason", lambda: "llm_provider_not_configured"
    )()
    return {
        "enabled": is_llm_writer and not reason,
        "provider": settings.llm_provider or "none",
        "model": settings.llm_model if settings.llm_enabled else "",
        "reason": reason,
    }


def uptime_seconds() -> int:
    """Whole seconds since the process started."""
    return int(time.monotonic() - _STARTED_MONOTONIC)


def reset_state() -> None:
    """Wipe all runtime state. Used by `/v1/teardown` and by tests."""
    _context_store.clear()
    _suppression_ledger.clear()
    _conversation_store.clear()
