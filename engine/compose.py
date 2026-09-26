"""`compose()` — the public entry point of the decision engine.

Pipeline, in order:

    normalize -> evaluate trigger -> select primary signal -> select supporting
    fact -> select offer -> select action + CTA -> sender identity ->
    suppression key -> MessagePlan -> render body -> validate -> result

Every stage is deterministic: no clock is read unless the caller passes `now`,
no randomness, no I/O, no model call. The same inputs always produce the same
`ComposedMessage`.

Wording is the one stage that may be delegated. Pass a `writer` and the body is
generated from the plan's `GenerationBrief`, validated against it, and replaced
by the deterministic rendering if anything fails. With no writer — the default —
`compose()` performs no I/O at all and stays byte-for-byte reproducible.

What a writer can change: how the body reads.
What it can never change: the trigger, the signal, the offer, the action, the
CTA, the sender, the suppression key, the rationale, or whether to send.
"""

from __future__ import annotations

import logging
from typing import Any

from engine import policy, render
from engine.actions import (
    audience_for,
    make_conversation_id,
    make_suppression_key,
    make_template_name,
    select_action,
    select_send_as,
)
from engine.brief import brief_for_plan
from engine.normalize import (
    normalize_category,
    normalize_customer,
    normalize_merchant,
    normalize_trigger,
)
from engine.offers import select_offer
from engine.signals import select_primary_signal, select_supporting_fact
from engine.triggers import evaluate_trigger
from engine.types import (
    ComposedMessage,
    GenerationBrief,
    MessagePlan,
    MessageWriter,
    NormalizedCategory,
    NormalizedCustomer,
    NormalizedMerchant,
    NormalizedTrigger,
    TriggerEvaluation,
)

logger = logging.getLogger("vera.compose")


def _no_action(
    reason_code: str,
    note: str,
    trigger: NormalizedTrigger | None,
    merchant: NormalizedMerchant | None,
    customer: NormalizedCustomer | None,
) -> ComposedMessage:
    """A reasoned decision not to send. Restraint is a valid outcome."""
    return ComposedMessage(
        decision="no_action",
        reason_code=reason_code,
        rationale=f"No message sent: {note}." if note else f"No message sent ({reason_code}).",
        merchant_id=merchant.merchant_id if merchant else "",
        customer_id=customer.customer_id if customer else None,
        trigger_id=trigger.id if trigger else "",
        suppression_key=trigger.suppression_key if trigger else "",
    )


def build_plan(
    category: NormalizedCategory,
    merchant: NormalizedMerchant,
    trigger: NormalizedTrigger,
    customer: NormalizedCustomer | None,
    evaluation: TriggerEvaluation,
    now: Any = None,
) -> MessagePlan | None:
    """Assemble the fact-only plan for one message.

    Returns None when no verifiable fact is available, which is the one case the
    later stages cannot recover from.
    """
    primary = select_primary_signal(category, merchant, trigger, customer, now)
    if primary is None:
        return None

    supporting = select_supporting_fact(
        category, merchant, trigger, customer, primary, now
    )
    offer = select_offer(category, merchant, trigger, customer)
    if offer is not None and not offer.is_existing and trigger.scope == "customer":
        # A catalog pattern is a proposal to the merchant, not something a
        # customer can take up — it is not live for this business.
        offer = None
    action = select_action(trigger, merchant, customer, primary, offer)
    send_as = select_send_as(trigger, customer)
    audience = audience_for(send_as)

    language = (
        render.language_for_customer(customer)
        if audience == "customer" and customer is not None
        else render.language_for_merchant(merchant)
    )
    # A message anchored entirely on category-level facts has nothing in it that is
    # specific to this business beyond the owner's name; the locality fills that gap.
    anchored_on_merchant = any(
        "merchant." in fact.source or "customer." in fact.source
        for fact in (primary, supporting) if fact
    )
    salutation = (
        render.customer_salutation(customer, merchant, language)
        if audience == "customer" and customer is not None
        else render.merchant_salutation(
            category, merchant, with_locality=not anchored_on_merchant
        )
    )

    citations = tuple(
        fact.citation for fact in (primary, supporting) if fact and fact.citation
    )
    evidence = tuple(fact.source for fact in (primary, supporting) if fact)
    if offer is not None:
        evidence += (
            "merchant.offers" if offer.is_existing else "category.offer_catalog",
        )

    return MessagePlan(
        merchant_id=merchant.merchant_id,
        category_slug=category.slug,
        trigger_id=trigger.id,
        trigger_kind=trigger.kind,
        family=policy.family_for(trigger.kind),
        urgency=trigger.urgency,
        priority=evaluation.priority,
        reason=evaluation.note or trigger.kind.replace("_", " "),
        primary_fact=primary,
        supporting_fact=supporting,
        offer=offer,
        action=action,
        cta=action.cta,
        send_as=send_as,
        audience=audience,
        salutation=salutation,
        language=language,
        voice_tone=category.voice.tone,
        suppression_key=make_suppression_key(trigger, merchant, customer, action),
        conversation_id=make_conversation_id(trigger, merchant, customer),
        template_name=make_template_name(trigger, send_as),
        customer_id=customer.customer_id if customer else None,
        citations=citations,
        evidence=evidence,
        taboo_terms=category.voice.vocab_taboo,
    )


def build_rationale(plan: MessagePlan) -> str:
    """Explain the decision, not the message.

    The judge cross-checks the rationale against the body, so this records what
    was chosen and why — the signal, its provenance, the offer's origin and the
    single ask — rather than restating the copy.
    """
    parts = [
        f"Chose {plan.trigger_kind} (urgency {plan.urgency}, priority {plan.priority:.2f}) "
        f"as the strongest reason to speak now"
    ]
    parts.append(
        f"anchored on {plan.primary_fact.kind} from {plan.primary_fact.source}"
    )
    if plan.supporting_fact:
        parts.append(
            f"paired with {plan.supporting_fact.kind} from {plan.supporting_fact.source} "
            f"to tie it to this {'customer' if plan.audience == 'customer' else 'merchant'}"
        )
    if plan.offer is not None:
        origin = (
            "the merchant's own active offer"
            if plan.offer.is_existing
            else "a category catalog pattern, offered as a proposal since no offer is live"
        )
        parts.append(f"used {plan.offer.title} ({origin})")
    else:
        parts.append("no offer referenced — none relevant was available to use")
    parts.append(
        f"single {plan.cta} ask ({plan.action.key}) sent as {plan.send_as} "
        f"in {plan.voice_tone or 'category'} voice"
    )
    if plan.citations:
        parts.append(f"cited {plan.citations[0]}")
    return "; ".join(parts) + "."


def write_body(
    brief: GenerationBrief,
    writer: MessageWriter | None,
) -> tuple[str, str]:
    """Produce the body for a brief, deterministically unless a writer is given.

    Returns `(body, source)`. A writer's output is validated against the brief
    before it is accepted, so an ungrounded, multi-ask or off-voice generation
    resolves to the deterministic rendering rather than going out.
    """
    # Imported here, not at module scope: the validator lives in services/ (where
    # the wording layer it guards lives) and imports `engine.types`, so a
    # top-level import would close a cycle through the engine package __init__.
    from services.message_validator import validate_generated_body

    if writer is None:
        return brief.deterministic_body, "deterministic"
    try:
        generated = writer.write(brief)
    except Exception as exc:  # pragma: no cover - writers swallow their own errors
        logger.warning("writer raised %s; using deterministic wording", type(exc).__name__)
        return brief.deterministic_body, "deterministic"

    if not generated.used_llm or not generated.body:
        return brief.deterministic_body, "deterministic"

    problems = validate_generated_body(generated.body, brief)
    if problems:
        logger.info("generated body rejected (%s)", ", ".join(problems[:3]))
        return brief.deterministic_body, "deterministic"
    return generated.body.strip(), "llm"


def compose(
    category: dict[str, Any] | None,
    merchant: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    customer: dict[str, Any] | None = None,
    now: Any = None,
    writer: MessageWriter | None = None,
) -> ComposedMessage:
    """Decide what Vera should say for one (category, merchant, trigger[, customer]).

    Args:
        category: CategoryContext payload, as stored from `/v1/context`.
        merchant: MerchantContext payload.
        trigger: TriggerContext payload — supplies the "why now".
        customer: CustomerContext payload, for customer-scoped triggers only.
        now: reference time for expiry and elapsed-day reasoning. Optional by
            design: without it the engine skips time-dependent judgements rather
            than reading the clock.
        writer: optional wording layer. Absent, the deterministic renderer
            writes the body and the call does no I/O.

    Returns:
        A `ComposedMessage`, which is either a send (with body, CTA, sender,
        suppression key and rationale) or a reasoned no-action.
    """
    norm_category = normalize_category(category)
    norm_merchant = normalize_merchant(merchant)
    norm_trigger = normalize_trigger(trigger)
    norm_customer = normalize_customer(customer)

    if norm_trigger is None:
        return _no_action(
            policy.REASON_MISSING_CONTEXT, "no trigger context supplied", None, norm_merchant, None
        )
    if norm_merchant is None or norm_category is None:
        missing = "merchant" if norm_merchant is None else "category"
        return _no_action(
            policy.REASON_MISSING_CONTEXT,
            f"{missing} context missing",
            norm_trigger,
            norm_merchant,
            norm_customer,
        )

    # A customer context belonging to a different merchant is not this
    # conversation's customer; ignore it rather than address the wrong person.
    if (
        norm_customer is not None
        and norm_customer.merchant_id
        and norm_merchant.merchant_id
        and norm_customer.merchant_id != norm_merchant.merchant_id
    ):
        norm_customer = None

    evaluation = evaluate_trigger(
        norm_category, norm_merchant, norm_trigger, norm_customer, now
    )
    if not evaluation.eligible:
        return _no_action(
            evaluation.reason_code, evaluation.note, norm_trigger, norm_merchant, norm_customer
        )

    plan = build_plan(
        norm_category, norm_merchant, norm_trigger, norm_customer, evaluation, now
    )
    if plan is None:
        return _no_action(
            policy.REASON_NO_FACT,
            f"no verifiable fact available for {norm_trigger.kind}",
            norm_trigger,
            norm_merchant,
            norm_customer,
        )

    brief = brief_for_plan(plan, norm_category, norm_merchant, norm_customer)
    _, params = render.render_body(plan, norm_category, norm_customer)
    body, body_source = write_body(brief, writer)
    problems = render.validate_body(body, plan)
    if problems:
        # Never send a body that would be penalised; the decision downgrades to
        # no-action and names the problem so it is visible rather than silent.
        return _no_action(
            "rendered_body_rejected",
            f"rendered body failed validation ({', '.join(problems)})",
            norm_trigger,
            norm_merchant,
            norm_customer,
        )

    return ComposedMessage(
        decision="send",
        reason_code=policy.REASON_SENT,
        rationale=build_rationale(plan),
        body=body,
        cta=plan.cta,
        send_as=plan.send_as,
        suppression_key=plan.suppression_key,
        template_name=plan.template_name,
        template_params=params,
        conversation_id=plan.conversation_id,
        merchant_id=plan.merchant_id,
        customer_id=plan.customer_id,
        trigger_id=plan.trigger_id,
        plan=plan,
        brief=brief,
        body_source=body_source,
    )


def compose_dict(
    category: dict[str, Any] | None,
    merchant: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    customer: dict[str, Any] | None = None,
    now: Any = None,
    writer: MessageWriter | None = None,
) -> dict[str, Any]:
    """`compose()` in the dict shape of challenge-brief.md §7.1."""
    return compose(category, merchant, trigger, customer, now, writer).to_dict()
