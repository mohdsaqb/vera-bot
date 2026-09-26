"""Action, CTA, sender identity, suppression key and conversation id.

One primary action per message, with a CTA drawn from the taxonomy the API
examples use (`open_ended`, `binary_yes_no`, `binary_confirm_cancel`,
`multi_choice_slot`, `none`). Multiple competing asks are an explicit
anti-pattern, so `select_action` returns exactly one `Action` and the renderer
puts its `ask` in the final sentence.
"""

from __future__ import annotations

import re

from engine import policy
from engine.types import (
    Action,
    NormalizedCustomer,
    NormalizedMerchant,
    NormalizedTrigger,
    OfferChoice,
    Signal,
)

SEND_AS_VERA = "vera"
SEND_AS_MERCHANT = "merchant_on_behalf"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------- #
# Sender identity
# --------------------------------------------------------------------------- #
def select_send_as(trigger: NormalizedTrigger, customer: NormalizedCustomer | None) -> str:
    """`merchant_on_behalf` for customer-scoped sends, `vera` otherwise.

    A customer-facing message goes out from the merchant's number and must read
    as the business talking to its own customer; everything else is Vera talking
    to the merchant.
    """
    if trigger.scope == "customer" and customer is not None:
        return SEND_AS_MERCHANT
    return SEND_AS_VERA


def audience_for(send_as: str) -> str:
    return "customer" if send_as == SEND_AS_MERCHANT else "merchant"


# --------------------------------------------------------------------------- #
# Action
# --------------------------------------------------------------------------- #
def _slots_available(primary: Signal) -> bool:
    slots = primary.data.get("slots")
    return isinstance(slots, list) and bool(slots)


def _preferred_slot_phrase(customer: NormalizedCustomer | None) -> str:
    """Describe the customer's recorded slot preference, or stay generic.

    `preferences.preferred_slots` holds values like "weekday_evening" or
    "saturday_afternoon"; using it is personalisation from supplied data, and its
    absence falls back to "a slot" rather than inventing a time.
    """
    if customer is None or not customer.preferred_slots:
        return "a slot"
    return f"a {customer.preferred_slots.replace('_', ' ')} slot"


def select_action(
    trigger: NormalizedTrigger,
    merchant: NormalizedMerchant,
    customer: NormalizedCustomer | None,
    primary: Signal,
    offer: OfferChoice | None,
) -> Action:
    """The single next step to ask for, chosen from the trigger family.

    Customer-facing families ask the customer to take up something concrete
    (a slot, a dispatch); merchant-facing families ask the merchant to approve
    work Vera has already framed, which is the "effort externalization" lever.
    Every ask names its object: a bare "want me to draft it?" gives the reader
    nothing to picture.
    """
    family = policy.family_for(trigger.kind)
    category = merchant.category_slug

    # ---- customer-facing ----
    if family in {"recall", "appointment", "trial", "occasion_customer"}:
        if _slots_available(primary):
            return Action(
                key="book_slot",
                cta=policy.CTA_SLOTS,
                ask="Reply with the slot that suits you, or tell us a time that works.",
                ask_hi="Jo slot theek lage reply kar dijiye, ya apna time bata dijiye.",
            )
        # With no slots supplied, the customer's own recorded preference is the
        # most useful thing to offer to hold.
        slot = _preferred_slot_phrase(customer)
        return Action(
            key="offer_booking",
            cta=policy.CTA_BINARY,
            ask=f"Reply YES and we will hold {slot} for you.",
            ask_hi=f"YES reply kijiye, hum {slot} rakh dete hain.",
        )

    if family == "refill":
        return Action(
            key="confirm_dispatch",
            cta=policy.CTA_CONFIRM,
            ask="Reply CONFIRM to send it out, or tell us if the dose has changed.",
            ask_hi="CONFIRM reply kijiye, ya dose badli ho to bata dijiye.",
        )

    if family == "winback":
        ask_en, ask_hi = policy.WINBACK_ASK_BY_CATEGORY.get(
            category,
            ("Reply YES and we will keep a spot for you — no commitment.",
             "YES reply kijiye — koi commitment nahi."),
        )
        return Action(key="offer_return", cta=policy.CTA_BINARY, ask=ask_en, ask_hi=ask_hi)

    # ---- merchant-facing ----
    if family == "curiosity":
        return Action(
            key="ask_question",
            cta=policy.CTA_OPEN,
            ask="What has been most asked-for this week?",
            ask_hi="Is hafte sabse zyada kya poocha ja raha hai?",
            effort_note="tell me and I'll turn it into a Google post for you",
            effort_note_hi="bata dijiye, main usska Google post bana deti hoon",
        )

    if family == "planning":
        topic = trigger.payload.get("intent_topic")
        what = str(topic).replace("_", " ") if isinstance(topic, str) and topic else "it"
        return Action(
            key="draft_content",
            cta=policy.CTA_BINARY,
            ask=f"Want me to draft the {what} so you can edit it?",
            ask_hi=f"Main {what} ka draft bana dun, aap edit kar lijiye?",
            effort_note=policy.EFFORT_NOTES["draft_content"],
            effort_note_hi="do minute mein draft ready hoga",
        )

    if family == "compliance":
        if trigger.kind == "supply_alert" and merchant.customer_aggregate:
            return Action(
                key="pull_list",
                cta=policy.CTA_BINARY,
                ask="Want me to pull the customers who bought those batches and draft their note?",
                ask_hi="Main un batches ke customers ki list nikaal ke unka note draft kar dun?",
                effort_note=policy.EFFORT_NOTES["pull_list"],
                effort_note_hi="list thodi der mein ready kar deti hoon",
            )
        return Action(
            key="compliance_checklist",
            cta=policy.CTA_BINARY,
            ask="Want me to put a checklist and a reminder against the deadline for you?",
            ask_hi="Main deadline ke against checklist aur reminder laga dun?",
        )

    if family == "knowledge":
        digest_kind = str(primary.data.get("kind", ""))
        if digest_kind == "cde":
            return Action(
                key="book_reminder",
                cta=policy.CTA_BINARY,
                ask="Want me to hold the date and send you the joining details?",
                ask_hi="Main date block kar ke joining details bhej dun?",
            )
        if digest_kind == "trend":
            return Action(
                key="draft_content",
                cta=policy.CTA_BINARY,
                ask="Want me to work that into your listing description?",
                ask_hi="Main aapki listing description mein yeh add kar dun?",
                effort_note=policy.EFFORT_NOTES["draft_content"],
            )
        return Action(
            key="draft_content",
            cta=policy.CTA_BINARY,
            ask="Want me to pull the full item and draft a patient note you can share?",
            ask_hi="Main poora item nikaal ke patient note draft kar dun?",
            effort_note=policy.EFFORT_NOTES["draft_content"],
            effort_note_hi="do minute mein draft ready hoga",
        )

    if family == "listing":
        return Action(
            key="verify_listing",
            cta=policy.CTA_BINARY,
            ask="Want me to start the verification?",
            ask_hi="Main verification shuru kar dun?",
            effort_note=policy.EFFORT_NOTES["verify_listing"],
            effort_note_hi="aapke end pe 5 minute ka kaam hai",
        )

    if family == "account":
        return Action(
            key="renew",
            cta=policy.CTA_BINARY,
            ask="Want me to set the renewal up?",
            ask_hi="Main renewal set kar dun?",
        )

    # A quiet listing with no live offer has one obvious first move.
    if offer is not None and not offer.is_existing:
        return Action(
            key="activate_offer",
            cta=policy.CTA_BINARY,
            ask=f"Want me to put {offer.title} live?",
            ask_hi=f"Main {offer.title} live kar dun?",
            effort_note=policy.EFFORT_NOTES["activate_offer"],
            effort_note_hi="set karne mein ek minute lagta hai",
        )

    # Concrete draft asks per family, each naming the artefact.
    draft_asks: dict[str, tuple[str, str]] = {
        "performance": (
            "Want me to draft a post to push while this is moving?",
            "Main ek post draft kar dun jab tak yeh chal raha hai?",
        ),
        "competition": (
            "Want me to draft a post that leads with what you do differently?",
            "Main ek post draft kar dun jo aapki khaas baat aage rakhe?",
        ),
        "occasion": (
            "Want me to draft the promo for it?",
            "Main iska promo draft kar dun?",
        ),
        "market": (
            "Want me to draft a shelf-and-post plan around that shift?",
            "Main iske hisaab se shelf aur post ka plan draft kar dun?",
        ),
        "milestone": (
            "Want me to draft the post for the day you cross it?",
            "Main us din ke liye post draft kar dun jab aap cross karenge?",
        ),
        "reputation": (
            "Want me to draft a reply you can use on those reviews?",
            "Main un reviews ke liye reply draft kar dun?",
        ),
        "reengagement": (
            "Want me to draft a post to get your listing moving again?",
            "Main ek post draft kar dun taaki listing phir chale?",
        ),
    }
    if family in draft_asks:
        ask_en, ask_hi = draft_asks[family]
        return Action(
            key="draft_content",
            cta=policy.CTA_BINARY,
            ask=ask_en,
            ask_hi=ask_hi,
            effort_note=policy.EFFORT_NOTES["draft_content"],
            effort_note_hi="do minute mein draft ready hoga",
        )

    return Action(
        key="ask_question",
        cta=policy.CTA_OPEN,
        ask="Want me to take this one from here?",
        ask_hi="Main isse aage dekh loon?",
    )


# --------------------------------------------------------------------------- #
# Suppression + conversation identity
# --------------------------------------------------------------------------- #
def _slug(value: str, limit: int = 28) -> str:
    """Lowercase, punctuation-free token safe for keys and ids.

    Trimmed after truncation too, so a cut that lands on a separator does not
    leave a dangling underscore.
    """
    cleaned = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return cleaned[:limit].strip("_") or "unknown"


def _time_bucket(trigger: NormalizedTrigger) -> str:
    """A stable bucket for the trigger's own story.

    Prefers a date the payload already carries so the same event always lands in
    the same bucket; falls back to the trigger id, which is unique per event.
    """
    for key in ("due_date", "date", "deadline_iso", "stock_runs_out_iso", "last_refill"):
        value = trigger.payload.get(key)
        if isinstance(value, str) and len(value) >= 7:
            return value[:7]
    return _slug(trigger.id, 36)


def make_suppression_key(
    trigger: NormalizedTrigger,
    merchant: NormalizedMerchant,
    customer: NormalizedCustomer | None,
    action: Action,
) -> str:
    """Deterministic dedup key for this logical outbound message.

    When the trigger supplies its own `suppression_key` that value is used
    verbatim: the judge generates it to identify the event, the API examples echo
    it back on the action, and re-deriving it would risk two keys for one story.

    When it is absent the key is synthesised from the dimensions that make a
    story distinct, family, recipient, action and a time bucket, so the same
    logical message always produces the same key, and two genuinely different
    events never collide.
    """
    if trigger.suppression_key:
        return trigger.suppression_key

    family = policy.family_for(trigger.kind)
    recipient = customer.customer_id if customer is not None else merchant.merchant_id
    return ":".join(
        (_slug(family, 20), _slug(recipient, 40), _slug(action.key, 20), _time_bucket(trigger))
    )


_ID_CORE_RE = re.compile(r"^([mc]_\d+)")


def _id_core(identifier: str) -> str:
    """Short, stable core of a dataset id: "m_019_karim_salon_lucknow" -> "m_019"."""
    match = _ID_CORE_RE.match(identifier.strip().lower())
    return match.group(1) if match else _slug(identifier, 16)


def make_conversation_id(
    trigger: NormalizedTrigger,
    merchant: NormalizedMerchant,
    customer: NormalizedCustomer | None,
) -> str:
    """Readable, stable conversation id.

    The case studies note that a decodable id (`conv_priya_recall_2026_11`) is
    worth more than an opaque one. Derived only from the ids involved, so the
    same story always resumes the same conversation.
    """
    family = policy.family_for(trigger.kind)
    parts = ["conv", _id_core(merchant.merchant_id)]
    if customer is not None:
        parts.append(_id_core(customer.customer_id))
    parts.extend((_slug(family, 20), _slug(_time_bucket(trigger), 24)))
    return "_".join(parts)


def make_template_name(trigger: NormalizedTrigger, send_as: str) -> str:
    """Pre-approved template name for the first touch in a session window.

    Follows the naming in the API examples: `vera_<kind>_v1` for merchant-facing,
    `merchant_<kind>_v1` for a send from the merchant's own number.
    """
    prefix = "merchant" if send_as == SEND_AS_MERCHANT else "vera"
    return f"{prefix}_{_slug(trigger.kind, 32)}_v1"
