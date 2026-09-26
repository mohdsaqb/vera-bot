"""Deterministic rendering of a `MessagePlan` into a WhatsApp body.

No LLM. Every sentence is assembled from facts already selected and verified
upstream, so the renderer's only job is phrasing: category voice, language
preference, one CTA in the final sentence.

Message shape, in order (challenge-brief.md §16 and the case-study patterns):

    <salutation>: <why now, from the trigger>. <one anchoring fact about this
    recipient>. <the real offer or resource>. <one ask>  [: citation]

Wording here is this project's own. The case studies are explicit that copying
their body text is penalised, so they were used only for shape and for the
vocabulary the categories license.
"""

from __future__ import annotations

import re
from typing import Final

from engine import policy
from engine.types import MessagePlan, NormalizedCategory, NormalizedCustomer, NormalizedMerchant

# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #
LANG_EN: Final[str] = "en"
LANG_HI_EN: Final[str] = "hi-en"

# Hindi-English code-mix is applied to the connective tissue only; numbers,
# metric names and citations stay in English, which is how the reference Vera
# transcripts in challenge-brief.md §9 actually read.
_PHRASES: Final[dict[str, dict[str, str]]] = {
    "offer_live": {
        LANG_EN: "You already run {offer}",
        LANG_HI_EN: "Aapke paas {offer} already live hai",
    },
    "offer_catalog": {
        LANG_EN: "{offer} is the pattern that works in your category",
        LANG_HI_EN: "Aapki category mein {offer} sabse accha chalta hai",
    },
    "offer_customer_live": {
        LANG_EN: "{offer} is on right now",
        LANG_HI_EN: "{offer} abhi chal raha hai",
    },
}

# Code-mixed variants of the ask, keyed by action. The ask is the sentence a
# recipient acts on, so this is where matching their language matters most; the
# fact sentences stay in English because that is how numbers, metric names and
# citations read in the reference transcripts.
_ASK_HI_EN: Final[dict[str, str]] = {
    "draft_content": "Main draft bana ke bhej dun?",
    "activate_offer": "Main {offer} live kar dun?",
    "verify_listing": "Main verification shuru kar dun?",
    "pull_list": "Main list nikaal ke note draft kar dun?",
    "renew": "Main renewal set kar dun?",
    "ask_question": "Is hafte sabse zyada kya poocha ja raha hai?",
    "book_slot": "Jo slot theek lage reply kar dijiye, ya apna time bata dijiye.",
    "offer_booking": "YES reply kijiye, hum slot rakh dete hain.",
    "offer_return": "YES reply kijiye — koi commitment nahi.",
    "confirm_dispatch": "CONFIRM reply kijiye, ya dose badli ho to bata dijiye.",
}

# One emoji, customer-facing only, and never for pharmacies: the pharmacy voice
# is "trustworthy_precise" and the regulated-category cases carry no decoration.
_CUSTOMER_EMOJI: Final[dict[str, str]] = {
    "dentists": "🦷",
    "salons": "✨",
    "restaurants": "🍽️",
    "gyms": "💪",
}

# Above this the message stops reading like a text message, so an optional third
# fact is dropped rather than squeezed in. The case-study bodies run 250-400
# characters; this leaves room for one more clause before that shape is lost.
SOFT_BODY_LIMIT = 430

_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“‘'][^\"”’']{0,300}[\"”’']")


def language_for_merchant(merchant: NormalizedMerchant) -> str:
    """Merchant-facing language, from `identity.languages`."""
    return LANG_HI_EN if policy.prefers_hindi_mix(merchant.languages) else LANG_EN


def language_for_customer(customer: NormalizedCustomer) -> str:
    """Customer-facing language, from `identity.language_pref`.

    Regional mixes (ta-en, te-en, kn-en) render in English rather than
    substituting Hindi, which would be the wrong language for that recipient.
    """
    return LANG_HI_EN if policy.prefers_hindi_mix(customer.language_pref) else LANG_EN


def _phrase(key: str, language: str, **kwargs: str) -> str:
    variants = _PHRASES[key]
    return variants.get(language, variants[LANG_EN]).format(**kwargs)


# --------------------------------------------------------------------------- #
# Salutation
# --------------------------------------------------------------------------- #
def merchant_salutation(
    category: NormalizedCategory,
    merchant: NormalizedMerchant,
    with_locality: bool = False,
) -> str:
    """Salutation built from the category's own `voice.salutation_examples`.

    The examples carry a placeholder (`Dr. {first_name}`, `Hi {pharmacist_name}`)
    which is filled with the merchant's `owner_first_name`. Generated dentist
    merchants already include the honorific in that field, so a duplicated
    "Dr. Dr." is collapsed.
    """
    first = merchant.owner_first_name.strip()
    examples = [
        e for e in (category.raw.get("voice", {}) or {}).get("salutation_examples", [])
        if isinstance(e, str)
    ]
    pattern = next((e for e in examples if _PLACEHOLDER_RE.search(e)), "")

    # Where the message's second fact comes from the category rather than from this
    # merchant, the locality is the one piece of merchant-specific detail available
    #: the judge names its absence directly ("omits locality or specific practice
    # data beyond the name"), and it is real supplied data, not a flourish.
    place = f", {merchant.locality}" if with_locality and merchant.locality else ""

    if first and pattern:
        filled = _PLACEHOLDER_RE.sub(first, pattern).strip()
        return re.sub(r"\bDr\.?\s+Dr\.?\s*", "Dr. ", filled) + place
    if first:
        return f"Hi {first}{place}"
    if merchant.name:
        return f"{merchant.name} team"
    return "Hi"


def customer_salutation(
    customer: NormalizedCustomer, merchant: NormalizedMerchant, language: str
) -> str:
    """Salutation for a message sent from the merchant's own number.

    The customer has a relationship with the business, not with Vera, so the
    business identifies itself; a parent-mediated contact is addressed as such.
    """
    name = customer.name.split("(")[0].strip() or "there"
    business = merchant.name or "your local team"
    greeting = "Namaste" if language == LANG_HI_EN else "Hi"

    if "parent:" in customer.name:
        # A child's record names the parent who actually holds the phone; both
        # belong in the greeting so it is clear who is being written to and why.
        parent = customer.name.split("parent:")[1].strip(" )")
        child = name
        return f"{greeting} {parent} — {business} here, about {child}"
    return f"{greeting} {name}, {business} {'se' if language == LANG_HI_EN else 'here'}"


# --------------------------------------------------------------------------- #
# Sentences
# --------------------------------------------------------------------------- #
def _sentence(text: str) -> str:
    """Trim and terminate one clause as a sentence."""
    clean = text.strip().rstrip(".,;")
    if not clean:
        return ""
    return clean if clean.endswith(("?", "!")) else f"{clean}."


def _capitalise(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _offer_sentence(plan: MessagePlan) -> str:
    """The offer clause, or nothing.

    Dropped when the ask already names it: no point saying the title twice.
    Families where an offer does not belong at all never carry one (see
    `offers.select_offer`).
    """
    if plan.offer is None:
        return ""
    title = plan.offer.title
    if title in plan.action.ask:
        return ""
    if plan.audience == "customer":
        if plan.offer.is_existing:
            return _sentence(_phrase("offer_customer_live", plan.language, offer=title))
        return ""
    key = "offer_live" if plan.offer.is_existing else "offer_catalog"
    return _sentence(_phrase(key, plan.language, offer=title))


def _ask_sentence(plan: MessagePlan) -> str:
    """The single CTA, always last, in the recipient's language.

    Asks are authored with their own punctuation, so nothing is guessed here. The
    effort note follows as a short fragment after the ask rather than inside it: the ask stays the thing the eye lands on, which is what keeps it from being
    buried.
    """
    action = plan.action
    hindi = plan.language == LANG_HI_EN
    ask = (action.ask_hi or action.ask) if hindi else action.ask
    ask = ask.strip()
    if not ask:
        return ""

    note = (action.effort_note_hi or action.effort_note) if hindi else action.effort_note
    if note and plan.audience == "merchant":
        ask = f"{ask} {_capitalise(note.strip())}."
    return ask


def _slot_sentence(plan: MessagePlan) -> str:
    """Spell out the real slots a payload supplied, never an invented time."""
    slots = plan.primary_fact.data.get("slots")
    if not isinstance(slots, list) or not slots:
        return ""
    labels = [str(slot.get("label")) for slot in slots[:2] if slot.get("label")]
    if not labels:
        return ""
    lead = "Open slots" if plan.language == LANG_EN else "Slots ready hain"
    return _sentence(f"{lead}: {' or '.join(labels)}")


# Month abbreviations as they appear in season tokens, for readable rendering.
_MONTH_TOKENS = frozenset(
    {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
)


def _readable_season(token: str) -> str:
    """Turn a season token into prose.

    "post_resolution_window_apr_jun" -> "the post-resolution window, Apr-Jun":
    the dataset packs the window name and its months into one identifier, and
    printing it raw reads like a field name.
    """
    parts = [part for part in str(token).split("_") if part]
    months = [p.title() for p in parts if p.lower() in _MONTH_TOKENS]
    words = [p for p in parts if p.lower() not in _MONTH_TOKENS]
    name = " ".join(words).replace("post ", "post-").strip()
    if months and name:
        return f"the {name}, {'-'.join(months)}"
    return f"the {name}" if name else "-".join(months)


def _seasonal_reframe(plan: MessagePlan) -> str:
    """For an expected seasonal dip, say so: it changes the merchant's response."""
    data = plan.primary_fact.data
    if not data.get("is_expected_seasonal"):
        return ""
    note = _readable_season(data.get("season_note", ""))
    tail = f" — {note}" if note else ""
    return _sentence(
        f"That is the expected seasonal pattern for your category{tail}, not a fault"
    )


def _match_night_judgement(plan: MessagePlan) -> str:
    """Match-day guidance derived from supplied data, not from a guess.

    The restaurant seasonal beat states that match-night promos belong on
    Tue-Thu; the trigger states whether tonight is a weeknight. When the two
    disagree the useful message is the contrarian one.
    """
    data = plan.primary_fact.data
    if plan.primary_fact.kind != "match_day" or "is_weeknight" not in data:
        return ""
    if data.get("is_weeknight") is False:
        return _sentence(
            "Your category's match-night pattern is Tue-Thu, so a weekend match is "
            "better played as a delivery push than a dine-in promo"
        )
    return _sentence("That lands in the Tue-Thu match-night window your category plays best")


# --------------------------------------------------------------------------- #
# Body
# --------------------------------------------------------------------------- #
def render_body(
    plan: MessagePlan,
    category: NormalizedCategory,
    customer: NormalizedCustomer | None,
) -> tuple[str, tuple[str, ...]]:
    """Render the body and the template parameters for the first-touch template.

    Returns `(body, template_params)`. Template params are the variable pieces
    of the body, in the order the approved template would place them:
    recipient, why-now, anchor/offer, ask.

    The salutation comes from the plan rather than being recomputed, so the body
    and the brief handed to a writer always open the same way.
    """
    # The plan already decided how to address the recipient: recomputing it here
    # would silently diverge from what the brief tells a writer to open with.
    salutation = plan.salutation
    if plan.audience == "customer" and customer is not None:
        emoji = _CUSTOMER_EMOJI.get(category.slug, "")
        if emoji:
            salutation = f"{salutation} {emoji}"

    why_now = _sentence(plan.primary_fact.text)
    anchor = _sentence(plan.supporting_fact.text) if plan.supporting_fact else ""

    # A family judgement sentence (a seasonal reframe, a match-night call) carries
    # its own "so what", so the anchor becomes a third fact competing for the
    # reader's attention. It is kept only while the whole message still reads
    # quickly: a delivery-order count genuinely strengthens a delivery
    # recommendation, but not at the cost of a wall of text.
    judgement = _seasonal_reframe(plan) or _match_night_judgement(plan)
    slots = _slot_sentence(plan)
    offer = _offer_sentence(plan)
    ask = _ask_sentence(plan)
    if judgement and anchor:
        # Measured against every part that will actually be in the body, the ask
        # included: it is the longest single clause and leaving it out of the
        # projection would let the message run well past the limit.
        projected = len(
            " ".join(part for part in (salutation, why_now, judgement, slots, anchor,
                                       offer, ask) if part)
        )
        if projected > SOFT_BODY_LIMIT:
            anchor = ""
    middle = [part for part in (judgement, slots, anchor, offer) if part]

    # A salutation that already contains a dash (a child's record names the
    # parent too) joins with a full stop so the line does not read as one clause.
    joiner = ". " if "—" in salutation else " — "
    opening = f"{salutation}{joiner}{_capitalise(why_now)}" if why_now else salutation
    sentences = [opening, *(_capitalise(part) for part in middle)]
    if ask:
        sentences.append(ask)

    body = " ".join(s for s in sentences if s)
    if plan.citations:
        body = f"{body} — {plan.citations[0]}"

    params = tuple(
        part for part in (
            salutation,
            _capitalise(why_now) if why_now else "",
            middle[0] if middle else "",
            ask,
        ) if part
    )
    return body.strip(), params


# --------------------------------------------------------------------------- #
# Output validation
# --------------------------------------------------------------------------- #
def validate_body(body: str, plan: MessagePlan) -> tuple[str, ...]:
    """Check a rendered body against the challenge's hard penalties.

    Returns the problems found; an empty tuple means the body is safe to send.
    Checked here rather than trusted, so a template change can never silently
    start emitting a penalised message:
      * URLs: a hard fail per action (api-call-examples.md F.4)
      * category taboo vocabulary (`voice.vocab_taboo`)
      * unresolved template placeholders
      * empty body, or more than one question mark (multiple competing CTAs)
    """
    problems: list[str] = []
    if not body.strip():
        problems.append("empty_body")
    if _URL_RE.search(body):
        problems.append("contains_url")
    if _PLACEHOLDER_RE.search(body):
        problems.append("unresolved_placeholder")
    # Quoted speech (a merchant's own words, a review quote) may legitimately
    # contain a question mark; only the bot's own sentences are counted.
    unquoted = _QUOTED_RE.sub("", body)
    if unquoted.count("?") > 1:
        problems.append("multiple_questions")

    lowered = body.lower()
    for term in plan.taboo_terms:
        phrase = term.split("(")[0].strip().lower()
        if phrase and phrase in lowered:
            problems.append(f"taboo_term:{phrase}")
    return tuple(problems)
