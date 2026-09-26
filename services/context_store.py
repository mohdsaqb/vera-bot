"""In-memory, versioned store for the four context scopes.

Owns *storage* semantics only — which version wins, what is currently held, how
many contexts exist per scope. It deliberately contains no decision logic
(trigger ranking, suppression, composition); those live in the Phase 2
engine and read from this store.

Version rules (challenge-testing-brief.md §2.1):
  * first version for a key           -> stored
  * strictly higher version           -> replaces the previous one atomically
  * same version, identical payload   -> already held; reported as success
  * same version, different payload   -> conflict, nothing overwritten
  * lower version                     -> ignored, never overwrites newer context

The two same-version cases are split deliberately. The brief calls a repeat
"idempotent", the API examples show a `409` for one, and the warmup check treats
any `accepted: false` as a failed warmup that disqualifies the bot for that slot —
so a retried push of *identical* content has to read as success, because nothing
about the stored state differs from what the caller asked for. A same-version push
carrying *different* content is a real disagreement about what that version means,
and that still conflicts.

Keys are ``(scope, context_id)``, so a merchant and a trigger may share an id
without colliding, and two scopes never interfere.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from models import SCOPES

STALE_VERSION = "stale_version"
ALREADY_STORED = "already_stored"


@dataclass(frozen=True)
class StoredContext:
    """One context object as currently held by the store."""

    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    stored_at: datetime
    delivered_at: str | None = None


@dataclass(frozen=True)
class SaveResult:
    """Outcome of a save attempt, with everything the API layer needs.

    Attributes:
        stored: True when this push became the held version.
        version: the version held *after* the call (the pushed one when
            ``stored``, otherwise the version already in the store).
        reason: why nothing was written — ``already_stored`` for an identical
            repeat, ``stale_version`` for a genuine conflict.
        record: the context now held for the key.
    """

    stored: bool
    version: int
    record: StoredContext
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        """True when the store holds what the caller asked it to hold.

        An identical repeat counts: the request has been satisfied, just not by
        this particular call.
        """
        return self.stored or self.reason == ALREADY_STORED


class ContextStore:
    """Thread-safe versioned context store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[tuple[str, str], StoredContext] = {}

    # ---------------------------------------------------------------- writes
    def save(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: dict[str, Any],
        delivered_at: str | None = None,
    ) -> SaveResult:
        """Apply the version rules and store the payload when it wins.

        Returns a :class:`SaveResult` describing what happened; never raises on
        a stale push.
        """
        key = (scope, context_id)
        with self._lock:
            current = self._items.get(key)
            if current is not None and version == current.version:
                identical = current.payload == payload
                return SaveResult(
                    stored=False,
                    version=current.version,
                    record=current,
                    reason=ALREADY_STORED if identical else STALE_VERSION,
                )
            if current is not None and version < current.version:
                return SaveResult(
                    stored=False,
                    version=current.version,
                    record=current,
                    reason=STALE_VERSION,
                )

            record = StoredContext(
                scope=scope,
                context_id=context_id,
                version=version,
                payload=payload,
                stored_at=datetime.now(timezone.utc),
                delivered_at=delivered_at,
            )
            self._items[key] = record
            return SaveResult(stored=True, version=version, record=record)

    def clear(self) -> int:
        """Drop all contexts and return how many were removed.

        Used by the optional `/v1/teardown` endpoint and by tests.
        """
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    # ----------------------------------------------------------------- reads
    def get(self, scope: str, context_id: str) -> StoredContext | None:
        """Return the held context for a key, or None."""
        with self._lock:
            return self._items.get((scope, context_id))

    def get_payload(self, scope: str, context_id: str) -> dict[str, Any] | None:
        """Return just the payload for a key, or None. Convenience for callers."""
        record = self.get(scope, context_id)
        return record.payload if record else None

    def get_version(self, scope: str, context_id: str) -> int | None:
        """Return the held version for a key, or None when unknown."""
        record = self.get(scope, context_id)
        return record.version if record else None

    def is_stale(self, scope: str, context_id: str, version: int) -> bool:
        """True when ``version`` would not win against what is already held."""
        current = self.get_version(scope, context_id)
        return current is not None and version <= current

    def list_scope(self, scope: str) -> list[StoredContext]:
        """Return every held context for one scope, ordered by context_id."""
        with self._lock:
            records = [r for (s, _), r in self._items.items() if s == scope]
        return sorted(records, key=lambda r: r.context_id)

    def counts(self) -> dict[str, int]:
        """Per-scope counts, zero-filled for all four known scopes.

        Unknown scopes cannot enter the store (the API validates first), so the
        result always matches the `contexts_loaded` shape in `/v1/healthz`.
        """
        counts = dict.fromkeys(SCOPES, 0)
        with self._lock:
            for scope, _ in self._items:
                counts[scope] = counts.get(scope, 0) + 1
        return counts

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
