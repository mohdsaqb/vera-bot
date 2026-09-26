"""Build the generation brief from a decision that has already been made.

This module is the boundary of what a writer may see. A `MessagePlan` holds the
whole decision — trigger ranking, provenance paths, suppression key, sender
identity — and most of that is none of a writer's business. `brief_for_plan`
projects out only the parts that belong in prose: the facts, the one ask, the
voice, and the deterministic rendering to fall back to.

Keeping the projection here (rather than handing the plan over whole) is what
makes the grounding check meaningful: the validator checks the body against the
brief, so a fact that never entered the brief cannot legitimately appear.
"""

from __future__ import annotations

from engine import render
from engine.types import (
    GenerationBrief,
    MessagePlan,
    NormalizedCategory,
    NormalizedCustomer,
    NormalizedMerchant,
)

# Voice guidance per category, in the writer's terms. Sourced from each
# category's `voice.tone`/`register` in the dataset, restated as instructions.
_STYLE_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "dentists": (
        "Write peer to peer, clinician to clinician.",
        "Clinical vocabulary is welcome; promotional language is not.",
        "Never imply a cure, a guarantee or a medical claim.",
    ),
    "salons": (
        "Warm and practical, like a colleague who knows the trade.",
        "Concrete services and prices beat adjectives.",
    ),
    "restaurants": (
        "Operator to operator. Short, busy, practical.",
        "Trade terms like covers and footfall are fine.",
    ),
    "gyms": (
        "Coach-like and direct, without hype.",
        "Never promise a physical result.",
    ),
    "pharmacies": (
        "Precise and trustworthy. Understate rather than overstate.",
        "Never give medical advice or imply a health claim.",
    ),
}

_AUDIENCE_NOTES: dict[str, tuple[str, ...]] = {
    "merchant": (
        "You are Vera, writing to the business owner.",
        "Do not re-introduce yourself.",
    ),
    "customer": (
        "You are writing as the business, to one of its own customers.",
        "Never mention the business's own analytics, rankings or benchmarks.",
        "Never mention Vera or any assistant.",
    ),
}

_LANGUAGE_NOTES: dict[str, str] = {
    render.LANG_EN: "Write in English.",
    render.LANG_HI_EN: (
        "Write in natural Hindi-English code-mix, the way Indian merchants "
        "actually text, in Latin script — never Devanagari. State the facts in "
        "English: numbers, metric names, dates, regulations, proper nouns and "
        "citations stay exactly as given, in an English sentence. Hindi carries "
        "the ask and the connective tissue, not the facts."
    ),
}


def _style_notes(plan: MessagePlan) -> tuple[str, ...]:
    """Assemble the writing constraints for this plan."""
    notes: list[str] = []
    notes.extend(_AUDIENCE_NOTES.get(plan.audience, ()))
    notes.extend(_STYLE_BY_CATEGORY.get(plan.category_slug, ()))
    notes.append(_LANGUAGE_NOTES.get(plan.language, _LANGUAGE_NOTES[render.LANG_EN]))
    if plan.offer is not None and not plan.offer.is_live_for_recipient:
        notes.append(
            "The offer named is a suggestion, not something already running — "
            "do not describe it as live."
        )
    if plan.cta == "none":
        notes.append("This message asks for nothing. Do not add a question.")
    return tuple(notes)


def brief_for_plan(
    plan: MessagePlan,
    category: NormalizedCategory,
    merchant: NormalizedMerchant,
    customer: NormalizedCustomer | None,
) -> GenerationBrief:
    """Project a `MessagePlan` into the brief a writer receives."""
    body, _ = render.render_body(plan, category, customer)
    recipient = (
        customer.name.split("(")[0].strip()
        if plan.audience == "customer" and customer is not None
        else merchant.owner_first_name or merchant.name
    )
    return GenerationBrief(
        purpose="outbound",
        category=category.display_name or category.slug,
        audience=plan.audience,
        recipient_name=recipient,
        salutation=plan.salutation,
        tone=plan.voice_tone,
        language=plan.language,
        reason=plan.trigger_kind.replace("_", " "),
        facts=tuple(fact.text for fact in plan.facts),
        offer_title=plan.offer.title if plan.offer else "",
        offer_is_live=bool(plan.offer and plan.offer.is_existing),
        ask=plan.action.ask_hi if plan.language == render.LANG_HI_EN and plan.action.ask_hi
        else plan.action.ask,
        cta=plan.cta,
        citation=plan.citations[0] if plan.citations else "",
        effort_note=plan.action.effort_note if plan.audience == "merchant" else "",
        taboo_terms=plan.taboo_terms,
        style_notes=_style_notes(plan),
        deterministic_body=body,
    )
