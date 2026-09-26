"""Vera's deterministic decision engine.

Given the four challenge contexts, decides whether to speak, what the strongest
factual reason is, which real offer to use, the single next step to ask for, and
what to do about each inbound reply: with no randomness, no clock reads and no
model calls. Wording may be delegated to a `MessageWriter`; nothing else can be.

Outbound:
    compose(category, merchant, trigger, customer=None, now=None, writer=None)
    compose_dict(...)                     the dict shape of challenge-brief §7.1
    build_plan(...) / MessagePlan         the decision, in full
    brief_for_plan(...) / GenerationBrief what a writer is allowed to see
    write_body(brief, writer)             generate, validate, fall back
    evaluate_trigger(...) / rank_triggers(...)
    select_primary_signal(...) / select_offer(...) / select_action(...)

Inbound:
    classify_reply(message, ...) -> IntentResult
    decide_reply(state, message, result, ...) -> ReplyDecision
    brief_for_reply(state, decision, message) -> GenerationBrief
"""

from engine.brief import brief_for_plan
from engine.compose import (
    build_plan,
    build_rationale,
    compose,
    compose_dict,
    write_body,
)
from engine.intent import INTENTS, IntentResult, classify_reply, looks_like_auto_reply
from engine.normalize import (
    normalize_category,
    normalize_customer,
    normalize_merchant,
    normalize_trigger,
)
from engine.offers import select_offer
from engine.reply import ReplyDecision, brief_for_reply, decide_reply, re_qualifies
from engine.signals import select_primary_signal, select_supporting_fact
from engine.triggers import evaluate_trigger, rank_triggers
from engine.types import (
    Action,
    ComposedMessage,
    ConversationState,
    GenerationBrief,
    GenerationResult,
    MessagePlan,
    MessageWriter,
    NormalizedCategory,
    NormalizedCustomer,
    NormalizedMerchant,
    NormalizedTrigger,
    OfferChoice,
    ReplyOutcome,
    Signal,
    TriggerEvaluation,
    Turn,
)

__all__ = [
    "INTENTS",
    "Action",
    "ComposedMessage",
    "ConversationState",
    "GenerationBrief",
    "GenerationResult",
    "IntentResult",
    "MessagePlan",
    "MessageWriter",
    "NormalizedCategory",
    "NormalizedCustomer",
    "NormalizedMerchant",
    "NormalizedTrigger",
    "OfferChoice",
    "ReplyDecision",
    "ReplyOutcome",
    "Signal",
    "TriggerEvaluation",
    "Turn",
    "brief_for_plan",
    "brief_for_reply",
    "build_plan",
    "build_rationale",
    "classify_reply",
    "compose",
    "compose_dict",
    "decide_reply",
    "evaluate_trigger",
    "looks_like_auto_reply",
    "normalize_category",
    "normalize_customer",
    "normalize_merchant",
    "normalize_trigger",
    "rank_triggers",
    "select_offer",
    "re_qualifies",
    "select_primary_signal",
    "select_supporting_fact",
    "write_body",
]
