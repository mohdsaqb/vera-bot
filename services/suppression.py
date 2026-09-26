"""Suppression ledger — what has already been said, and against which context.

Three guards, all required by the challenge's penalty table and its adaptive
scoring:

* one send per logical story (`suppression_key` is the dedup dimension the
  trigger itself defines), so calling `/v1/tick` again does not retell it;
* no repeated body inside a conversation, which carries an explicit
  anti-repetition penalty (challenge-testing-brief.md §10);
* but a story *may* be told again when the context it was built from has been
  refreshed and the message actually changed. The judge injects updated
  performance snapshots and new digest items mid-test and scores bots on whether
  later sends reflect them, so a key that blocked forever would score as
  "ignored the new context".

Those three combine into one rule (`should_send`): the same story goes out again
only when a newer context version has arrived *and* the message it produces is
different. Neither condition alone is enough — a version bump that changes
nothing stays quiet, and a reworded body with no new facts stays quiet too.

Kept out of `ContextStore` on purpose: the store holds what the judge pushed,
this holds what the bot did. Both are process-level state and both reset for
tests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from hashlib import blake2s

# Reasons `should_send` returns, for logs and tests.
FIRST_SEND = "first_send"
CONTEXT_REFRESHED = "context_refreshed"
DUPLICATE_BODY = "duplicate_body"
STORY_ALREADY_SENT = "story_already_sent"


def _fingerprint(text: str) -> str:
    """Short stable hash of a message body, for repeat detection."""
    return blake2s(" ".join(text.split()).lower().encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True)
class SentRecord:
    """What went out under one suppression key."""

    conversation_id: str
    body_hash: str
    context_version: int


class SuppressionLedger:
    """Records the stories already told and the bodies already sent."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._keys: dict[str, SentRecord] = {}
        self._bodies: dict[str, set[str]] = {}

    # ----------------------------------------------------------------- checks
    def should_send(
        self, suppression_key: str, body: str, context_version: int = 0
    ) -> tuple[bool, str]:
        """Decide whether this story may go out, and say why.

        Args:
            suppression_key: identifies the logical story.
            body: the message as it would be sent.
            context_version: a monotonic stamp for the contexts the message was
                built from. A higher value than the last send under this key means
                the judge has pushed something newer.
        """
        if not suppression_key:
            return True, FIRST_SEND
        with self._lock:
            previous = self._keys.get(suppression_key)
            if previous is None:
                return True, FIRST_SEND
            if previous.body_hash == _fingerprint(body):
                return False, DUPLICATE_BODY
            if context_version > previous.context_version:
                return True, CONTEXT_REFRESHED
            return False, STORY_ALREADY_SENT

    def is_suppressed(self, suppression_key: str) -> bool:
        """True when anything has gone out under this key.

        Used for flags that are set rather than sent — an opt-out marker, say —
        where there is no body to compare.
        """
        if not suppression_key:
            return False
        with self._lock:
            return suppression_key in self._keys

    def is_repeat(self, conversation_id: str, body: str) -> bool:
        """True when this exact body was already sent in this conversation."""
        if not conversation_id or not body:
            return False
        with self._lock:
            return _fingerprint(body) in self._bodies.get(conversation_id, set())

    def context_version_for(self, suppression_key: str) -> int | None:
        """The context version the last send under this key was built from."""
        with self._lock:
            record = self._keys.get(suppression_key)
            return record.context_version if record else None

    # ------------------------------------------------------------------ write
    def record(
        self,
        suppression_key: str,
        conversation_id: str,
        body: str,
        context_version: int = 0,
    ) -> None:
        """Note that a message went out, so the guards see it next time."""
        with self._lock:
            if suppression_key:
                self._keys[suppression_key] = SentRecord(
                    conversation_id=conversation_id,
                    body_hash=_fingerprint(body),
                    context_version=context_version,
                )
            if conversation_id and body:
                self._bodies.setdefault(conversation_id, set()).add(_fingerprint(body))

    def clear(self) -> None:
        with self._lock:
            self._keys.clear()
            self._bodies.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)
