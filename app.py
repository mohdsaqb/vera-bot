"""FastAPI application exposing the judge-facing API surface.

Five endpoints, per `challenge-testing-brief.md` §2:

    GET  /v1/healthz   liveness + how much context is loaded
    GET  /v1/metadata  bot identity
    POST /v1/context   versioned context ingestion
    POST /v1/tick      periodic wake-up; the bot may initiate conversations
    POST /v1/reply     a merchant/customer reply; the bot answers synchronously

plus the optional `POST /v1/teardown` (§11) which wipes stored context.

Handlers stay thin: request validation lives in `models.py`, storage semantics
in `services/context_store.py`, decisions in `engine/`, orchestration in
`services/`, configuration in `config.py`. No handler calls a model, and none of
them can fail because a model did.

Two safety nets wrap every route: an access log that records method, path, status
and latency without touching bodies or headers, and a catch-all handler that turns
any unhandled fault into valid JSON while logging the traceback server-side.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import get_settings
from models import (
    ContextAccepted,
    ContextCounts,
    ContextPush,
    ContextRejected,
    HealthResponse,
    MetadataResponse,
    ReplyRequest,
    ReplyResponse,
    TeardownResponse,
    TickAction,
    TickRequest,
    TickResponse,
)
from state import (
    STARTED_AT,
    get_context_store,
    get_decision_service,
    get_reply_service,
    llm_status,
    reset_state,
    uptime_seconds,
)

_settings = get_settings()

logging.basicConfig(
    level=getattr(logging, _settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("vera.api")

_LLM_STATUS = llm_status()
logger.info(
    "wording layer: %s (provider=%s model=%s%s)",
    "llm" if _LLM_STATUS["enabled"] else "deterministic",
    _LLM_STATUS["provider"], _LLM_STATUS["model"] or "-",
    f" reason={_LLM_STATUS['reason']}" if _LLM_STATUS["reason"] else "",
)

app = FastAPI(
    title="Vera Bot",
    description="magicpin AI Challenge — merchant engagement bot (foundation phase)",
    version=_settings.bot_version,
)


def _iso_utc_ms(moment: datetime) -> str:
    """Format a datetime as `2026-04-26T10:00:00.123Z`, the shape the judge uses."""
    utc = moment.astimezone(timezone.utc)
    return f"{utc.strftime('%Y-%m-%dT%H:%M:%S')}.{utc.microsecond // 1000:03d}Z"


def _rejected(response: ContextRejected, http_status: int) -> JSONResponse:
    """Serialise a rejection, omitting the fields that do not apply."""
    return JSONResponse(
        status_code=http_status,
        content=response.model_dump(exclude_none=True),
    )


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def _summarise_errors(errors: Sequence[Mapping[str, Any]]) -> str:
    """Field-level summary of validation failures.

    Only locations and messages are included: never the submitted values, so
    merchant/customer payload data can never leak into a response or a log.
    """
    parts = []
    for error in errors[:5]:
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        parts.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
    return "; ".join(parts) or "invalid request body"


# Dataset context payloads are a few KB, so anything far larger is a mistake or an
# attempt to exhaust a small instance. Checking the declared length rejects it
# before the body is buffered; a chunked request without Content-Length slips past
# this, which is proportionate for an endpoint reached only by the judge harness.
MAX_BODY_BYTES = 1_048_576


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request: method, path, status, latency.

    Deliberately narrow: no bodies, no headers, no query strings. Merchant and
    customer payloads pass through these endpoints and an access log is not the
    place for them; the identifiers that matter are already logged by the handlers
    that act on them.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        logger.warning("rejected %s %s: body %s bytes exceeds limit",
                       request.method, request.url.path, declared)
        return JSONResponse(
            status_code=413,
            content={"error": "payload_too_large",
                     "details": f"request body must be under {MAX_BODY_BYTES} bytes"},
        )
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        logger.exception(
            "%s %s failed after %.1fms", request.method, request.url.path, elapsed
        )
        raise
    elapsed = (time.perf_counter() - started) * 1000
    # Health polls every 60s for the whole window; logging them at info would
    # bury everything else.
    level = logging.DEBUG if request.url.path.endswith("/healthz") else logging.INFO
    logger.log(
        level, "%s %s -> %d in %.1fms",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: answer with valid JSON, log the cause, hide the internals.

    The judge scores a malformed response and penalises a timeout, so an
    unhandled fault must still produce a well-formed body. The traceback goes to
    the server log in full: this catches the response, it does not hide the bug.
    """
    logger.exception(
        "unhandled error on %s %s (%s)",
        request.method, request.url.path, type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "details": "the request could not be completed"},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return the contract's 400 shape instead of FastAPI's default 422.

    `/v1/context` has a documented malformed-request body
    (`{accepted: false, reason, details}`); other endpoints get a plain error
    object. A bad `scope` is reported as `invalid_scope` specifically.
    """
    errors = exc.errors()
    details = _summarise_errors(errors)

    if request.url.path.endswith("/v1/context"):
        mentions_scope = any("scope" in e.get("loc", ()) for e in errors)
        return _rejected(
            ContextRejected(
                reason="invalid_scope" if mentions_scope else "invalid_request",
                details=details,
            ),
            status.HTTP_400_BAD_REQUEST,
        )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "invalid_request", "details": details},
    )


# --------------------------------------------------------------------------- #
# GET /: signpost only
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root() -> dict[str, object]:
    """Name the service and list its endpoints.

    The contract defines nothing here and the judge appends each path to the base
    URL, so nothing depends on this. It exists because the submitted link is the
    base URL, and a bare 404 reads as a broken deployment to anyone who opens it.
    """
    return {
        "service": "vera",
        "status": "ok",
        "endpoints": [
            "GET  /v1/healthz",
            "GET  /v1/metadata",
            "POST /v1/context",
            "POST /v1/tick",
            "POST /v1/reply",
            "POST /v1/teardown",
        ],
    }


# --------------------------------------------------------------------------- #
# GET /v1/healthz
# --------------------------------------------------------------------------- #
@app.get("/v1/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness probe. Pure in-memory reads: no LLM, no database, no I/O.

    The judge polls this every 60s and compares `contexts_loaded` against what
    it pushed during warmup, so the counts must reflect the store exactly.
    """
    counts = get_context_store().counts()
    return HealthResponse(
        uptime_seconds=uptime_seconds(),
        contexts_loaded=ContextCounts(**counts),
    )


# --------------------------------------------------------------------------- #
# GET /v1/metadata
# --------------------------------------------------------------------------- #
@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    """Bot identity. Sourced from configuration; never exposes secrets."""
    settings = get_settings()
    return MetadataResponse(
        team_name=settings.team_name,
        team_members=settings.team_members,
        model=settings.reported_model,
        approach=settings.approach,
        contact_email=settings.contact_email,
        version=settings.bot_version,
        submitted_at=settings.submitted_at or _iso_utc_ms(STARTED_AT),
    )


# --------------------------------------------------------------------------- #
# POST /v1/context
# --------------------------------------------------------------------------- #
@app.post("/v1/context")
async def push_context(body: ContextPush) -> JSONResponse:
    """Ingest one versioned context object.

    Storage semantics live in the store; this handler only maps the outcome onto
    the wire contract:

      * 200 + ack when the push is stored, and also when it repeats a version
        already held with identical content: a retry has to read as success,
        since the judge's warmup check treats any `accepted: false` as a failed
        warmup;
      * 409 + `stale_version` when an older version arrives, or when the same
        version arrives carrying different content;
      * 400 when malformed (handled by :func:`handle_validation_error`).
    """
    result = get_context_store().save(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
        delivered_at=body.delivered_at,
    )

    if not result.accepted:
        logger.info(
            "context push conflicted: scope=%s id=%s version=%s held=%s reason=%s",
            body.scope, body.context_id, body.version, result.version, result.reason,
        )
        return _rejected(
            ContextRejected(reason=result.reason or "stale_version",
                            current_version=result.version),
            status.HTTP_409_CONFLICT,
        )

    logger.info(
        "context %s: scope=%s id=%s version=%s",
        "stored" if result.stored else "already held",
        body.scope, body.context_id, body.version,
    )
    accepted = ContextAccepted(
        ack_id=f"ack_{body.context_id}_v{body.version}",
        stored_at=_iso_utc_ms(result.record.stored_at),
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=accepted.model_dump())


# --------------------------------------------------------------------------- #
# POST /v1/tick
# --------------------------------------------------------------------------- #
@app.post("/v1/tick", response_model=TickResponse)
async def tick(body: TickRequest) -> TickResponse:
    """Periodic wake-up. Returns the proactive messages to send right now.

    The work is delegated: `DecisionService` resolves the available triggers
    against the stored contexts, the engine decides and composes, the wording
    layer words it, and the suppression ledger keeps a story from being told
    twice. Every action opens or continues a conversation so the reply that comes
    back has context. An empty list is a valid, and often correct, answer: the
    challenge rewards restraint.
    """
    actions = get_decision_service().actions(body.available_triggers, body.now)
    logger.info(
        "tick at %s: %d trigger(s) available, %d action(s) returned",
        body.now, len(body.available_triggers), len(actions),
    )
    return TickResponse(actions=[TickAction(**action) for action in actions])


# --------------------------------------------------------------------------- #
# POST /v1/reply
# --------------------------------------------------------------------------- #
@app.post("/v1/reply", response_model=ReplyResponse, response_model_exclude_none=True)
async def reply(body: ReplyRequest) -> ReplyResponse:
    """Handle an inbound merchant or customer turn.

    The intent is classified and the action decided deterministically
    (`engine/intent.py`, `engine/reply.py`); wording may be generated, and falls
    back to the deterministic draft on any problem. A conversation the bot never
    opened is treated as a new thread rather than an error: the judge's replay
    scenarios post into ids of their own.
    """
    outcome = get_reply_service().handle(
        body.conversation_id,
        body.message,
        merchant_id=body.merchant_id or "",
        customer_id=body.customer_id,
        from_role=body.from_role,
        received_at=body.received_at or "",
    )
    logger.info(
        "reply on conversation=%s from=%s turn=%s -> %s (%s)",
        body.conversation_id, body.from_role, body.turn_number,
        outcome.action, outcome.intent,
    )
    return ReplyResponse(
        action=outcome.action,
        body=outcome.body or None,
        cta=outcome.cta if outcome.action == "send" else None,
        wait_seconds=outcome.wait_seconds or None,
        rationale=outcome.rationale,
    )


# --------------------------------------------------------------------------- #
# POST /v1/teardown (optional, challenge-testing-brief.md §11)
# --------------------------------------------------------------------------- #
@app.post("/v1/teardown", response_model=TeardownResponse)
async def teardown() -> TeardownResponse:
    """Wipe all stored context at end of test, as the privacy rule requires."""
    cleared = len(get_context_store())
    reset_state()
    logger.info("teardown: cleared %d context(s)", cleared)
    return TeardownResponse(cleared=cleared)
