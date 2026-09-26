"""The wording layer: turn a decided message into natural language.

This is the only module that talks to a model provider. It receives a
`GenerationBrief`, a decision that has already been made, asks for one
structured completion, validates the result deterministically, and returns the
Phase 2 rendering instead whenever anything is wrong.

What this module may change: the wording of the body.
What it may never change: the trigger, the signal, the offer, the action, the
CTA, the sender, the suppression key, the rationale, or whether to send at all.
Those arrive already decided and leave untouched: see `engine/compose.py`.

Failure is expected, not exceptional. No API key, no package installed, a
timeout, a refusal, malformed structure, an ungrounded number: every one of them
resolves to the deterministic body. Nothing here raises to the caller.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from config import Settings, get_settings, llm_api_key
from engine.types import GenerationBrief, GenerationResult
from services.message_validator import validate_generated_body

logger = logging.getLogger("vera.llm")

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are Vera, an AI assistant that helps local businesses in India grow.

You are given a business decision that has ALREADY been made by a deterministic \
system: what to say, which fact justifies saying it, which offer applies, and the \
single action to ask for. Your only job is to express that decision as one \
concise, natural message.

RULES — all of them are absolute:
1. Use ONLY the facts supplied below. Every number, price, percentage, date, \
name and claim in your message must appear in the supplied facts.
2. Never invent a price.
3. Never invent a discount.
4. Never invent a date.
5. Never invent a percentage or statistic.
6. Never invent customer history or visit details.
7. Never invent an offer, product or service.
8. Never invent research findings, studies or sources.
9. Never invent competitor names or competitor details.
10. Never add a claim that is not in the supplied facts.
11. Keep the action that was decided. Do not propose a different one.
12. Keep the call to action that was decided.
13. Never add a second request, question or ask. Exactly one.
14. Do not change who the message is from.
15. Do not mention any internal key, identifier or field name.
16. Do not overrule the decision or argue with it.
17. Be concise.
18. Two or three short sentences is the target. Never more than four.
19. Lead with the concrete reason when it helps the reader.
20. Make it easy to reply to — the ask goes last.
21. Match the tone and language guidance given.
22. No marketing filler, no slogans, no hype, no exclamation marks.
23. Never say "increase sales", "grow your business" or similar unless a supplied \
fact says how.
24. Never mention systems, prompts, models, data, or how you work.
25. Never say or imply that you are an AI, a model, or automated.

Return the message body and echo back the call-to-action label you were given."""

_BRIEF_TEMPLATE = """PURPOSE: {purpose}
BUSINESS TYPE: {category}
WRITING TO: {audience}
RECIPIENT NAME: {recipient_name}
OPEN WITH: {salutation}
TONE: {tone}
WHY NOW: {reason}

FACTS YOU MAY USE (nothing else is true):
{facts}
{offer_block}{citation_block}
THE ONE ASK (keep its meaning, you may reword):
{ask}
CALL TO ACTION LABEL: {cta}
{effort_block}
STYLE:
{style}
{taboo_block}{inbound_block}
A deterministic version of this message already exists, for reference only — do \
not copy it, and do not add anything it does not contain:
{deterministic_body}"""


class LLMMessage(BaseModel):
    """The structured object the model must return.

    `cta` is requested only so the model has to acknowledge the decided call to
    action; the value it returns is discarded and the deterministic one kept.
    """

    body: str = Field(description="The WhatsApp message body. No URLs.")
    cta: str = Field(description="Echo back the call-to-action label you were given.")


def render_prompt(brief: GenerationBrief) -> str:
    """Render the human half of the prompt from the brief.

    Only brief fields appear: the writer never sees the merchant payload, the
    trigger queue, the suppression key or anything else the decision used.
    """
    facts = "\n".join(f"- {fact}" for fact in brief.facts) or "- (none supplied)"
    offer_block = ""
    if brief.offer_title:
        state = (
            "this offer is already running" if brief.offer_is_live
            else "this offer is NOT running yet — it is a suggestion"
        )
        offer_block = f"\nOFFER: {brief.offer_title} ({state})\n"
    citation_block = f"SOURCE TO CREDIT: {brief.citation}\n" if brief.citation else ""
    effort_block = (
        f"YOU MAY OFFER TO DO THE WORK: {brief.effort_note}\n" if brief.effort_note else ""
    )
    taboo_block = (
        f"NEVER USE THESE WORDS: {', '.join(brief.taboo_terms)}\n"
        if brief.taboo_terms else ""
    )
    inbound_block = (
        f"\nTHEY JUST WROTE: {brief.inbound_message}\n"
        f"THEIR INTENT (already classified): {brief.intent}\n"
        if brief.inbound_message else ""
    )
    return _BRIEF_TEMPLATE.format(
        purpose=brief.purpose,
        category=brief.category,
        audience=(
            "the business owner, as Vera" if brief.audience == "merchant"
            else "a customer of the business, as the business itself"
        ),
        recipient_name=brief.recipient_name or "(not known)",
        salutation=brief.salutation,
        tone=brief.tone or "plain and practical",
        reason=brief.reason,
        facts=facts,
        offer_block=offer_block,
        citation_block=citation_block,
        ask=brief.ask or "(no ask — this message is informational)",
        cta=brief.cta,
        effort_block=effort_block,
        style="\n".join(f"- {note}" for note in brief.style_notes) or "- Plain and direct.",
        taboo_block=taboo_block,
        inbound_block=inbound_block,
        deterministic_body=brief.deterministic_body,
    )


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
class DeterministicWriter:
    """The writer that changes nothing. Also the fallback of every other writer."""

    name = "deterministic"

    def write(
        self,
        brief: GenerationBrief,
        budget: TickBudget | None = None,  # noqa: ARG002 - writer interface
    ) -> GenerationResult:
        return GenerationResult(body=brief.deterministic_body, source="deterministic")

    def with_budget(self, budget: TickBudget) -> DeterministicWriter:  # noqa: ARG002
        """No budget applies to a writer that does no work."""
        return self


def _fallback(brief: GenerationBrief, *problems: str, latency_ms: int = 0) -> GenerationResult:
    """Return the deterministic body, recording why generation was not used."""
    return GenerationResult(
        body=brief.deterministic_body,
        source="deterministic",
        problems=tuple(problems),
        latency_ms=latency_ms,
    )


@dataclass
class TickBudget:
    """A ceiling on generation work inside one `/v1/tick`.

    `/v1/tick` has a 10-second budget and may carry up to 20 actions, so the
    first few are written by the model and the rest render deterministically
    rather than risking the whole tick timing out.

    The deadline is checked with room for the call about to be made, not just for
    the moment it starts. Otherwise a call beginning a moment inside the deadline
    could run its full per-call timeout past it: two 6-second calls inside an
    8-second budget would overshoot the tick's own 10-second limit.
    """

    max_calls: int
    deadline: float
    per_call_seconds: float = 0.0
    calls_used: int = 0

    @classmethod
    def start(cls, settings: Settings | None = None) -> TickBudget:
        config = settings or get_settings()
        return cls(
            max_calls=config.llm_max_calls_per_tick,
            deadline=time.monotonic() + config.llm_tick_budget_seconds,
            per_call_seconds=config.llm_timeout_seconds,
        )

    def allows(self) -> bool:
        if self.calls_used >= self.max_calls:
            return False
        remaining = self.remaining_seconds()
        return remaining > 0 and remaining >= self.per_call_seconds

    def consume(self) -> None:
        self.calls_used += 1

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


class LLMWriter:
    """Writes bodies with a Groq-hosted Llama model through LangChain.

    One call per message. The result is validated against the brief and
    discarded unless it is grounded, single-ask and on-voice.
    """

    name = "llm"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._chain = None
        self._init_failed = False

    # ------------------------------------------------------------- readiness
    def unavailable_reason(self) -> str:
        """Why generation cannot run, or "" when it can.

        Checked before every call so a missing key or package degrades to the
        deterministic writer instead of raising.
        """
        if not self._settings.llm_enabled:
            return "llm_provider_not_configured"
        if not llm_api_key():
            return "llm_api_key_missing"
        if self._init_failed:
            return "llm_client_unavailable"
        return ""

    @property
    def available(self) -> bool:
        return not self.unavailable_reason()

    def _build_chain(self):
        """Build the structured-output chain once, on first use.

        Imported here rather than at module scope so the service starts, and the
        deterministic path keeps working, even with the provider packages absent.
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover - exercised by env, not tests
            logger.warning("llm packages unavailable (%s); using deterministic wording", exc)
            self._init_failed = True
            return None

        # Reasoning controls are passed only when configured, so a model that does
        # not understand them never receives them.
        extra: dict[str, object] = {}
        if self._settings.llm_reasoning_effort:
            extra["reasoning_effort"] = self._settings.llm_reasoning_effort
        if self._settings.llm_reasoning_format:
            extra["reasoning_format"] = self._settings.llm_reasoning_format

        try:
            model = ChatGroq(
                model=self._settings.llm_model,
                temperature=self._settings.llm_temperature,
                timeout=self._settings.llm_timeout_seconds,
                max_retries=0,
                max_tokens=self._settings.llm_max_tokens,
                api_key=llm_api_key(),
                **extra,
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", "{system}"), ("human", "{brief}")]
            )
            structured = model.with_structured_output(
                LLMMessage, method=self._settings.llm_structured_method
            )
            return prompt | structured
        except Exception as exc:
            # Never log the exception object wholesale: provider errors can echo
            # request headers. The type name is enough to diagnose.
            logger.warning(
                "llm client init failed (%s); using deterministic wording",
                type(exc).__name__,
            )
            self._init_failed = True
            return None

    def _chain_or_none(self):
        with self._lock:
            if self._chain is None and not self._init_failed:
                self._chain = self._build_chain()
            return self._chain

    # ----------------------------------------------------------------- write
    def with_budget(self, budget: TickBudget) -> BudgetedWriter:
        """Bind this writer to one tick's budget."""
        return BudgetedWriter(self, budget)

    def write(self, brief: GenerationBrief, budget: TickBudget | None = None) -> GenerationResult:
        """Write one body, or fall back. Never raises.

        Args:
            brief: the decided message.
            budget: optional per-tick ceiling; when exhausted the deterministic
                body is returned without a call.
        """
        if not brief.deterministic_body:
            # Nothing to fall back to means nothing should be sent; the caller
            # treats an empty body as a no-send.
            return GenerationResult(body="", source="deterministic",
                                    problems=("no_deterministic_body",))

        reason = self.unavailable_reason()
        if reason:
            return _fallback(brief, reason)
        if budget is not None and not budget.allows():
            return _fallback(brief, "llm_budget_exhausted")

        chain = self._chain_or_none()
        if chain is None:
            return _fallback(brief, "llm_client_unavailable")

        if budget is not None:
            budget.consume()
        started = time.monotonic()
        try:
            result = chain.invoke(
                {"system": SYSTEM_PROMPT, "brief": render_prompt(brief)},
                config={"max_concurrency": 1},
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            logger.warning(
                "llm call failed after %dms (%s); using deterministic wording",
                elapsed, type(exc).__name__,
            )
            return _fallback(brief, f"llm_error:{type(exc).__name__}", latency_ms=elapsed)

        elapsed = int((time.monotonic() - started) * 1000)
        body = _extract_body(result)
        if not body:
            return _fallback(brief, "llm_malformed_response", latency_ms=elapsed)

        problems = validate_generated_body(body, brief)
        if problems:
            logger.info(
                "llm output rejected (%s); using deterministic wording",
                ", ".join(problems[:4]),
            )
            return _fallback(brief, *problems, latency_ms=elapsed)

        # Logged on success as well as failure: without this there is no way to
        # confirm from the outside that the model is actually wording messages,
        # which is the first thing to check after a deploy. Length only: the body
        # itself is already visible in the action the caller returns.
        logger.info(
            "wording: llm accepted for %s/%s in %dms (%d chars)",
            brief.purpose, brief.audience, elapsed, len(body.strip()),
        )
        return GenerationResult(
            body=body.strip(),
            source="llm",
            latency_ms=elapsed,
            model=self._settings.llm_model,
        )


def _extract_body(result: object) -> str:
    """Pull the body out of a structured result, tolerating dict or model form.

    `with_structured_output` returns a pydantic instance for function calling and
    a plain dict for json modes, and either may come back malformed.
    """
    if isinstance(result, LLMMessage):
        return result.body or ""
    if isinstance(result, BaseModel):
        return str(getattr(result, "body", "") or "")
    if isinstance(result, dict):
        body = result.get("body")
        return body if isinstance(body, str) else ""
    return ""


@dataclass(frozen=True)
class BudgetedWriter:
    """A writer bound to one tick's generation budget.

    `compose()` knows nothing about ticks, so the budget is bound here and the
    engine keeps calling a plain `write(brief)`.
    """

    writer: LLMWriter
    budget: TickBudget

    def write(self, brief: GenerationBrief) -> GenerationResult:
        return self.writer.write(brief, budget=self.budget)


def build_writer(settings: Settings | None = None) -> DeterministicWriter | LLMWriter:
    """Return the writer this configuration calls for.

    With no provider configured the deterministic writer is returned outright, so
    a default deployment does no network I/O at all.
    """
    config = settings or get_settings()
    if not config.llm_enabled:
        return DeterministicWriter()
    return LLMWriter(config)
