"""Typed values passed between decision-engine stages.

Everything here is a frozen dataclass: a stage takes normalized context in and
returns an immutable decision out, so the pipeline is easy to test in isolation
and impossible to mutate accidentally halfway through.

`MessagePlan` is the important one — it is the contract between this
deterministic engine and the future LLM renderer. It carries only facts that
were read out of the four contexts, each with its provenance, so a generator
consuming it cannot invent numbers it was not given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# --------------------------------------------------------------------------- #
# Normalized context
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Voice:
    """Category voice profile (`category.voice`)."""

    tone: str = ""
    register: str = ""
    code_mix: str = ""
    vocab_allowed: tuple[str, ...] = ()
    vocab_taboo: tuple[str, ...] = ()


@dataclass(frozen=True)
class OfferTemplate:
    """One entry of `category.offer_catalog` or `merchant.offers`."""

    id: str
    title: str
    value: str = ""
    audience: str = ""
    type: str = ""
    status: str = ""
    source: str = "category_catalog"  # or "merchant_offers"


@dataclass(frozen=True)
class DigestItem:
    """One entry of `category.digest`."""

    id: str
    kind: str = ""
    title: str = ""
    source: str = ""
    summary: str = ""
    actionable: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedCategory:
    slug: str
    display_name: str
    voice: Voice
    offer_catalog: tuple[OfferTemplate, ...] = ()
    peer_stats: dict[str, Any] = field(default_factory=dict)
    digest: tuple[DigestItem, ...] = ()
    content_library: tuple[dict[str, Any], ...] = ()
    seasonal_beats: tuple[dict[str, Any], ...] = ()
    trend_signals: tuple[dict[str, Any], ...] = ()
    regulators: tuple[str, ...] = ()
    journals: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    def digest_item(self, item_id: str | None) -> DigestItem | None:
        """Resolve a digest item by id (triggers reference items by id only)."""
        if not item_id:
            return None
        for item in self.digest:
            if item.id == item_id:
                return item
        return None

    def peer(self, key: str) -> float | None:
        value = self.peer_stats.get(key)
        return float(value) if isinstance(value, (int, float)) else None


@dataclass(frozen=True)
class NormalizedMerchant:
    merchant_id: str
    category_slug: str
    name: str
    owner_first_name: str
    city: str
    locality: str
    languages: tuple[str, ...] = ()
    verified: bool | None = None
    established_year: int | None = None

    subscription_status: str = ""
    plan: str = ""
    days_remaining: int | None = None
    days_since_expiry: int | None = None

    performance: dict[str, Any] = field(default_factory=dict)
    delta_7d: dict[str, Any] = field(default_factory=dict)
    offers: tuple[OfferTemplate, ...] = ()
    conversation_history: tuple[dict[str, Any], ...] = ()
    customer_aggregate: dict[str, Any] = field(default_factory=dict)
    signals: tuple[str, ...] = ()
    signal_values: dict[str, Any] = field(default_factory=dict)
    review_themes: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def active_offers(self) -> tuple[OfferTemplate, ...]:
        return tuple(o for o in self.offers if o.status == "active")

    @property
    def expired_offers(self) -> tuple[OfferTemplate, ...]:
        return tuple(o for o in self.offers if o.status and o.status != "active")

    @property
    def has_talked_before(self) -> bool:
        """True when Vera and this merchant already have a message history.

        Drives salutation choice — re-introducing yourself is a listed
        anti-pattern (challenge-brief.md §11).
        """
        return bool(self.conversation_history)

    def metric(self, key: str) -> float | None:
        value = self.performance.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    def delta(self, metric: str) -> float | None:
        value = self.delta_7d.get(f"{metric}_pct")
        return float(value) if isinstance(value, (int, float)) else None

    def aggregate(self, key: str) -> Any:
        return self.customer_aggregate.get(key)

    def has_signal(self, needle: str) -> bool:
        return any(needle in signal for signal in self.signals)


@dataclass(frozen=True)
class NormalizedCustomer:
    customer_id: str
    merchant_id: str
    name: str
    language_pref: str = ""
    age_band: str = ""
    is_senior: bool = False
    contactable: bool = True

    state: str = ""
    first_visit: str = ""
    last_visit: str = ""
    visits_total: int | None = None
    services_received: tuple[str, ...] = ()
    lifetime_value: float | None = None
    relationship_extra: dict[str, Any] = field(default_factory=dict)

    preferred_slots: str = ""
    channel: str = ""
    reminder_opt_in: bool | None = None
    preferences_extra: dict[str, Any] = field(default_factory=dict)

    opted_in_at: str = ""
    consent_scope: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_any_consent(self) -> bool:
        return bool(self.opted_in_at and self.consent_scope)


@dataclass(frozen=True)
class NormalizedTrigger:
    id: str
    kind: str
    scope: str
    source: str
    urgency: int
    merchant_id: str = ""
    customer_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    suppression_key: str = ""
    expires_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_placeholder(self) -> bool:
        """True for generated triggers carrying no factual payload.

        `generate_dataset.py` emits `{"placeholder": true, "metric_or_topic": …}`
        for the 75 expanded triggers, so the engine must fall back to
        merchant/category facts — or decline — rather than invent the missing ones.
        """
        return bool(self.payload.get("placeholder")) or not self.payload


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Signal:
    """One verifiable fact, with where it came from.

    `text` is merchant/customer-ready phrasing; `source` is the dotted path in
    the input contexts, which is what keeps the engine honest — a signal with no
    source cannot be built.
    """

    kind: str
    text: str
    strength: float
    source: str
    citation: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OfferChoice:
    """The single offer or resource to put in the message."""

    offer: OfferTemplate
    is_existing: bool
    reason: str

    @property
    def title(self) -> str:
        return self.offer.title

    @property
    def is_live_for_recipient(self) -> bool:
        """True only for an offer the merchant is actually running.

        A catalog pattern is a proposal; describing it as live would be a claim
        the merchant could check and disprove.
        """
        return self.is_existing


@dataclass(frozen=True)
class Action:
    """The one thing Vera asks for. `cta` is the challenge's CTA taxonomy.

    `ask` carries its own final punctuation — whether a given ask reads as a
    question or as an instruction is a property of how it was written, not
    something to infer later. `*_hi` are the Hindi-English variants, used when
    the recipient's language preference includes Hindi.
    """

    key: str
    cta: str
    ask: str
    ask_hi: str = ""
    effort_note: str = ""
    effort_note_hi: str = ""


@dataclass(frozen=True)
class TriggerEvaluation:
    """Why (or why not) this trigger justifies speaking now."""

    trigger_id: str
    eligible: bool
    priority: float
    components: dict[str, float] = field(default_factory=dict)
    reason_code: str = "eligible"
    note: str = ""


@dataclass(frozen=True)
class MessagePlan:
    """Deterministic, fact-only plan for one outbound message.

    This is the hand-off point for Phase 3: an LLM renderer should be able to
    write the body from this object alone, and anything not present here is
    something it must not claim.
    """

    merchant_id: str
    category_slug: str
    trigger_id: str
    trigger_kind: str
    family: str
    urgency: int
    priority: float

    reason: str
    primary_fact: Signal
    supporting_fact: Signal | None
    offer: OfferChoice | None
    action: Action
    cta: str

    send_as: str
    audience: str
    salutation: str
    language: str
    voice_tone: str

    suppression_key: str
    conversation_id: str
    template_name: str

    customer_id: str | None = None
    citations: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    taboo_terms: tuple[str, ...] = ()

    @property
    def facts(self) -> tuple[Signal, ...]:
        return tuple(f for f in (self.primary_fact, self.supporting_fact) if f)


# --------------------------------------------------------------------------- #
# Reply outcome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReplyOutcome:
    """What `/v1/reply` answers with.

    Mirrors the three actions the contract allows. `body`/`cta` are set only for
    a send; `wait_seconds` only for a wait.
    """

    action: str  # "send" | "wait" | "end"
    intent: str
    rationale: str
    body: str = ""
    cta: str = "none"
    wait_seconds: int = 0
    body_source: str = "deterministic"


# --------------------------------------------------------------------------- #
# Conversation state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Turn:
    """One message in a conversation."""

    direction: str  # "outbound" | "inbound"
    body: str
    at: str = ""
    intent: str = ""
    cta: str = ""


@dataclass(frozen=True)
class ConversationState:
    """Everything known about one thread.

    Frozen: every transition returns a new state, so a half-applied update
    cannot exist and a state handed to the decision layer cannot be mutated
    behind its back.
    """

    conversation_id: str
    merchant_id: str = ""
    customer_id: str | None = None

    # What this conversation is about — the decision that opened it.
    trigger_id: str = ""
    family: str = ""
    action_key: str = ""
    cta: str = "none"
    send_as: str = "vera"
    audience: str = "merchant"
    language: str = "en"
    category_slug: str = ""
    suppression_key: str = ""
    brief: GenerationBrief | None = None

    turns: tuple[Turn, ...] = ()
    turn_count: int = 0
    last_intent: str = ""
    active: bool = True
    should_end: bool = False
    end_reason: str = ""

    consecutive_auto_replies: int = 0
    unanswered_nudges: int = 0
    delay_requested_seconds: int = 0
    last_inbound_at: str = ""

    @property
    def outbound(self) -> tuple[Turn, ...]:
        return tuple(turn for turn in self.turns if turn.direction == "outbound")

    @property
    def inbound(self) -> tuple[Turn, ...]:
        return tuple(turn for turn in self.turns if turn.direction == "inbound")

    @property
    def last_outbound(self) -> Turn | None:
        outbound = self.outbound
        return outbound[-1] if outbound else None

    @property
    def inbound_bodies(self) -> tuple[str, ...]:
        return tuple(turn.body for turn in self.inbound)

    @property
    def expects_slot(self) -> bool:
        """True when the last thing sent asked the recipient to pick a slot."""
        last = self.last_outbound
        return bool(last and last.cta == "multi_choice_slot")

    @property
    def is_first_touch(self) -> bool:
        """True when nothing has gone out on this thread yet.

        The first outbound to a merchant or customer must use a pre-approved
        template (challenge-brief.md §5.1); later turns in the same session may be
        free-form. Because a conversation id is derived per story, every `/v1/tick`
        action is the first touch of its own thread and every `/v1/reply` answer is
        in-session.
        """
        return not self.outbound

    @property
    def session_state(self) -> str:
        """Where this thread stands, in the vocabulary of the session rules.

        Derived rather than stored, so it cannot drift from the turns:

            new            opened, nothing back yet
            active         they replied and the thread is open
            awaiting_reply we answered their reply and are waiting again
            holding        they asked for time
            ended          closed, by them or by us
        """
        if self.should_end or not self.active:
            return "ended"
        if self.delay_requested_seconds:
            return "holding"
        if not self.inbound:
            return "new"
        last = self.turns[-1] if self.turns else None
        return "awaiting_reply" if last and last.direction == "outbound" else "active"

    @property
    def has_real_engagement(self) -> bool:
        """True once a human has replied with something other than canned text."""
        return any(
            turn.intent not in {"", "auto_reply"} for turn in self.inbound
        )

    def already_sent(self, body: str) -> bool:
        """True when this exact body has gone out on this thread before.

        Repeating a body carries an anti-repetition penalty, so the reply layer
        checks this before sending.
        """
        needle = " ".join(body.split()).lower()
        return any(
            " ".join(turn.body.split()).lower() == needle for turn in self.outbound
        )


# --------------------------------------------------------------------------- #
# Generation hand-off
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenerationBrief:
    """Everything a writer is allowed to know, and nothing else.

    Built from a decision that has already been made. It carries the facts that
    may be stated, the one ask, and the voice to use — but not the trigger queue,
    the suppression key, the sender decision or any context the decision did not
    select. A writer cannot widen the message because it cannot see anything
    wider.

    `deterministic_body` is the Phase 2 rendering of the same brief: the
    fallback whenever generation is unavailable, slow or ungrounded.
    """

    purpose: str  # "outbound" | "reply"
    category: str
    audience: str  # "merchant" | "customer"
    recipient_name: str
    salutation: str
    tone: str
    language: str  # "en" | "hi-en"
    reason: str
    facts: tuple[str, ...]
    offer_title: str = ""
    offer_is_live: bool = False
    ask: str = ""
    cta: str = "none"
    citation: str = ""
    effort_note: str = ""
    taboo_terms: tuple[str, ...] = ()
    style_notes: tuple[str, ...] = ()
    deterministic_body: str = ""
    inbound_message: str = ""
    intent: str = ""

    @property
    def grounding_text(self) -> str:
        """Every string the body may draw facts from, concatenated.

        The validator checks generated numbers, amounts, dates and proper nouns
        against this, so anything absent here counts as invented.
        """
        parts = [
            *self.facts,
            self.offer_title,
            self.ask,
            self.salutation,
            self.recipient_name,
            self.citation,
            self.effort_note,
            self.deterministic_body,
            self.category,
        ]
        return " \n".join(part for part in parts if part)


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of asking a writer for a body."""

    body: str
    source: str  # "llm" | "deterministic"
    problems: tuple[str, ...] = ()
    latency_ms: int = 0
    model: str = ""

    @property
    def used_llm(self) -> bool:
        return self.source == "llm"


class MessageWriter(Protocol):
    """A thing that turns a brief into a body.

    The engine depends on this protocol, never on a provider SDK, so
    `engine/` stays free of network code and a fake writer is enough for tests.
    """

    def write(self, brief: GenerationBrief) -> GenerationResult:
        """Return a body for this brief, falling back to the deterministic one."""
        ...


@dataclass(frozen=True)
class ComposedMessage:
    """Result of `compose()` — either a message to send, or a reasoned no-op."""

    decision: str  # "send" | "no_action"
    reason_code: str
    rationale: str
    body: str = ""
    cta: str = "none"
    send_as: str = "vera"
    suppression_key: str = ""
    template_name: str = ""
    template_params: tuple[str, ...] = ()
    conversation_id: str = ""
    merchant_id: str = ""
    customer_id: str | None = None
    trigger_id: str = ""
    plan: MessagePlan | None = None
    brief: GenerationBrief | None = None
    body_source: str = "deterministic"

    @property
    def should_send(self) -> bool:
        return self.decision == "send"

    def to_dict(self) -> dict[str, Any]:
        """The composition contract from challenge-brief.md §7.1."""
        return {
            "body": self.body,
            "cta": self.cta,
            "send_as": self.send_as,
            "suppression_key": self.suppression_key,
            "rationale": self.rationale,
        }

    def to_action(self) -> dict[str, Any]:
        """The `/v1/tick` action shape from challenge-testing-brief.md §2.2."""
        return {
            "conversation_id": self.conversation_id,
            "merchant_id": self.merchant_id,
            "customer_id": self.customer_id,
            "send_as": self.send_as,
            "trigger_id": self.trigger_id,
            "template_name": self.template_name,
            "template_params": list(self.template_params),
            "body": self.body,
            "cta": self.cta,
            "suppression_key": self.suppression_key,
            "rationale": self.rationale,
        }
