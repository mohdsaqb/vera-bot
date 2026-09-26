"""Pydantic request/response models for the judge-facing HTTP contract.

Shapes come verbatim from `challenge-testing-brief.md` §2 and
`examples/api-call-examples.md`. Context payloads are kept as
``dict[str, Any]`` on purpose: the four scopes have different, evolving
shapes (the judge adds digest items and performance fields mid-test), so the
outer envelope is validated strictly while the inner payload is stored as-is.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

# The four context scopes the judge pushes. Anything else is a 400
# ("invalid_scope") per challenge-testing-brief.md §2.1.
Scope = Literal["category", "merchant", "customer", "trigger"]
SCOPES: tuple[str, ...] = get_args(Scope)

SendAs = Literal["vera", "merchant_on_behalf"]
ReplyAction = Literal["send", "wait", "end"]


# --------------------------------------------------------------------------- #
# POST /v1/context
# --------------------------------------------------------------------------- #
class ContextPush(BaseModel):
    """Body of `POST /v1/context`: one versioned context object."""

    # Tolerate extra envelope keys so a future judge field never 400s a push.
    model_config = ConfigDict(extra="allow")

    scope: Scope
    context_id: str = Field(min_length=1)
    version: int = Field(ge=0)
    payload: dict[str, Any]
    # Opaque ISO-8601 stamp from the judge; stored for audit, never parsed.
    delivered_at: str | None = None


class ContextAccepted(BaseModel):
    """200 response: the push was stored."""

    accepted: Literal[True] = True
    ack_id: str
    stored_at: str


class ContextRejected(BaseModel):
    """409/400 response: the push was not stored.

    `current_version` is set for stale pushes, `details` for malformed ones;
    both are omitted from the wire body when unset.
    """

    accepted: Literal[False] = False
    reason: str
    current_version: int | None = None
    details: str | None = None


# --------------------------------------------------------------------------- #
# GET /v1/healthz
# --------------------------------------------------------------------------- #
class ContextCounts(BaseModel):
    """Per-scope context counts. All four keys are always present."""

    category: int = 0
    merchant: int = 0
    customer: int = 0
    trigger: int = 0


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    uptime_seconds: int
    contexts_loaded: ContextCounts


# --------------------------------------------------------------------------- #
# GET /v1/metadata
# --------------------------------------------------------------------------- #
class MetadataResponse(BaseModel):
    team_name: str
    team_members: list[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str


# --------------------------------------------------------------------------- #
# POST /v1/tick
# --------------------------------------------------------------------------- #
class TickRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Simulated "now" from the judge; opaque ISO-8601 string.
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class TickAction(BaseModel):
    """One proactive outbound message.

    Every field here is required by the judge's action schema: omitting any of
    them costs a malformed-action penalty (see api-call-examples.md F.2).
    Phase 2 fills these in; Phase 1 never emits an action.
    """

    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    send_as: SendAs
    trigger_id: str
    template_name: str | None = None
    template_params: list[str] = Field(default_factory=list)
    body: str
    cta: str
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: list[TickAction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# POST /v1/reply
# --------------------------------------------------------------------------- #
class ReplyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    conversation_id: str = Field(min_length=1)
    merchant_id: str | None = None
    customer_id: str | None = None
    # "merchant" or "customer" in practice; left as a free string so an
    # unexpected role never rejects an otherwise-valid turn.
    from_role: str = Field(min_length=1)
    message: str
    received_at: str | None = None
    turn_number: int | None = None


class ReplyResponse(BaseModel):
    """Response to `POST /v1/reply`.

    `body`/`cta` are set for `action: "send"`, `wait_seconds` for
    `action: "wait"`; unused fields are omitted from the wire body.
    """

    action: ReplyAction
    body: str | None = None
    cta: str | None = None
    wait_seconds: int | None = None
    rationale: str


# --------------------------------------------------------------------------- #
# POST /v1/teardown (optional, challenge-testing-brief.md §11)
# --------------------------------------------------------------------------- #
class TeardownResponse(BaseModel):
    status: Literal["ok"] = "ok"
    cleared: int
