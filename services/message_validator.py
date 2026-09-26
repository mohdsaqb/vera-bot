"""Deterministic validation of a generated message body.

Nothing a writer produces is trusted. Every body is checked against the
`GenerationBrief` it was written from, and a body that fails any check is
discarded in favour of the deterministic rendering: the engine never repairs a
bad generation, and never asks a second model to grade the first.

The checks, in the order they run:

    empty / length          a body must exist and stay readable
    numbers                 every figure must appear in the brief
    amounts                 every ₹ amount must appear in the brief
    percentages             every % must appear in the brief
    dates                   every date-like token must appear in the brief
    proper nouns            no name, brand or place the brief does not contain
    offer identity          the offer named is the offer the decision chose
    internal identifiers    no snake_case field names, no system vocabulary
    single ask              one question, one request
    taboo vocabulary        the category's own forbidden words
    URLs                    a hard fail in the judge's penalty table
    audience leakage        a customer never sees the merchant's analytics
    offer state             a suggested offer is never called live

Grounding is checked by containment rather than by meaning: a figure is
acceptable only if that exact figure is in the brief. That is blunt, and it is
the point: it cannot be talked around.
"""

from __future__ import annotations

import re

from engine.types import GenerationBrief

# Bounds. The floor rejects a truncated generation; the ceiling keeps a body
# inside what the case studies treat as readable (theirs run 250-400 chars).
MIN_BODY_CHARS = 40
MAX_BODY_CHARS = 700

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_AMOUNT_RE = re.compile(r"₹\s?\d+(?:[.,]\d+)*")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“‘'][^\"”’']{0,300}[\"”’']")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_MONTHS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)
# "12 Nov", "Nov 12", "2026-12-15", "Wed 5 Nov"
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")[a-z]*"
    r"|(?:" + "|".join(_MONTHS) + r")[a-z]*\s+\d{1,2})\b",
    re.IGNORECASE,
)

# Requests. Two of these in one message is two asks.
_REQUEST_RE = re.compile(
    r"\b(?:want me to|shall i|should i|would you like me to|can i|may i|main\b)",
    re.IGNORECASE,
)
# Action verbs, used to catch one question carrying several asks
# ("activate it and update your listing?"). Only words that read as verbs in this
# domain: "post" is excluded because it is almost always the object: "draft a
# post", "post draft kar dun", and counting it would flag one action as two.
_ACTION_VERBS = (
    "activate", "update", "draft", "send", "publish", "launch", "change",
    "pull", "book", "verify", "renew", "schedule", "set up",
)

# Words that would tell a reader how the system works.
_INTERNAL_TERMS = (
    "messageplan", "suppression", "trigger_id", "context_id", "merchant_id",
    "customer_id", "primary_fact", "supporting_fact", "decision engine",
    "deterministic", "fallback", "language model", "large language model",
    "llm", "as an ai", "i am an ai", "prompt", "json", "schema", "token",
    "api", "payload", "dataset", "pipeline",
)

# Phrases that assert an offer is already running.
_LIVE_CLAIMS = (
    "already running", "already live", "is live", "currently running",
    "currently live", "is on right now", "already on", "abhi chal raha",
    "already live hai", "chal raha hai",
)

# Analytics a customer must never be shown.
_MERCHANT_ONLY_TERMS = (
    "click-through", "clickthrough", "ctr", "peer median", "peer benchmark",
    "profile views", "percentage points", "median", "benchmark", "listing",
)

# Capitalised words that are ordinary English rather than proper nouns. Checked
# before a capitalised token is treated as a name the brief should have had.
_COMMON_CAPITALS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "for",
        "from", "have", "here", "how", "i", "if", "in", "is", "it", "its",
        "just", "may", "me", "my", "no", "not", "of", "on", "one", "or", "our",
        "out", "reply", "shall", "should", "so", "that", "the", "their", "them",
        "then", "there", "they", "this", "to", "up", "want", "was", "we",
        "what", "when", "which", "who", "will", "with", "would", "yes", "you",
        "your", "yours", "confirm", "stop", "ok", "okay", "google", "whatsapp",
        "vera", "namaste", "hi", "hello", "quick", "worth", "open", "slots",
        "new", "next", "now", "today", "tomorrow", "week", "month",
        "main", "aap", "aapke", "aapki", "aapka", "apke", "hai", "hain", "kar",
        "dun", "dijiye", "kijiye", "bata", "koi", "nahi", "mein", "ke", "ki",
        "ka", "se", "bhej", "ready", "abhi", "chal", "raha", "rakh", "dete",
        "haan", "ya", "bana", "nikaal", "poora", "item", "patient", "note",
        "draft", "live", "set", "minute", "lagta", "takes", "tell", "turn",
        "post", "line", "answer", "enough", "jo", "theek", "lage",
        "apna", "time", "badli", "ho", "dose", "hafte", "sabse", "zyada",
        "kya", "poocha", "ja", "usska", "commitment", "do",
    }
)


def _normalise_number(token: str) -> str:
    """Strip thousands separators so 2,410 and 2410 compare equal."""
    return token.replace(",", "").rstrip(".")


def _numbers(text: str) -> set[str]:
    return {_normalise_number(match) for match in _NUMBER_RE.findall(text)}


def _amounts(text: str) -> set[str]:
    return {
        _normalise_number(match.replace("₹", "").strip())
        for match in _AMOUNT_RE.findall(text)
    }


def _percentages(text: str) -> set[str]:
    return {match.replace(" ", "").rstrip("%") for match in _PERCENT_RE.findall(text)}


def _dates(text: str) -> set[str]:
    return {match.lower().replace("  ", " ") for match in _DATE_RE.findall(text)}


_POSSESSIVE_RE = re.compile(r"['’]s$|s['’]$")
_FULL_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def _canonical_noun(word: str) -> str:
    """Fold a proper noun to the form the grounding text would hold.

    Two differences are spelling, not invention, and folding them prevents a
    faithful rewording from being thrown away:
      * a possessive, "JIDA's" is the "JIDA" the brief cited;
      * a month written out, "October" is the "Oct" in a source line.
    """
    stripped = _POSSESSIVE_RE.sub("", word).lower()
    if stripped in _FULL_MONTHS or stripped[:3] in _MONTHS:
        return stripped[:3]
    return stripped


def _proper_nouns(text: str) -> set[str]:
    """Capitalised tokens that are not sentence-initial and not common words."""
    found: set[str] = set()
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        words = sentence.split()
        for index, raw in enumerate(words):
            word = raw.strip(".,;:!?()\"'“”’—-")
            if len(word) < 3 or not word[0].isupper():
                continue
            if index == 0:  # sentence-initial capital carries no information
                continue
            if word.lower() in _COMMON_CAPITALS or word.isupper():
                continue
            found.add(word)
    return found


def _ask_sentence(body: str) -> str:
    """The sentence carrying the request, for the single-ask checks."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(body.strip()) if s.strip()]
    for sentence in reversed(sentences):
        if "?" in sentence or _REQUEST_RE.search(sentence):
            return sentence
    return sentences[-1] if sentences else body


def validate_generated_body(body: str, brief: GenerationBrief) -> tuple[str, ...]:
    """Check a generated body against its brief.

    Returns the problems found, most structural first; an empty tuple means the
    body is safe to send. Callers treat any problem as a reason to fall back.
    """
    problems: list[str] = []
    text = (body or "").strip()

    if not text:
        return ("empty_body",)
    if len(text) < MIN_BODY_CHARS:
        problems.append("body_too_short")
    if len(text) > MAX_BODY_CHARS:
        problems.append("body_too_long")
    if _URL_RE.search(text):
        problems.append("contains_url")

    allowed = brief.grounding_text

    problems.extend(
        f"unsupported_number:{n}" for n in sorted(_numbers(text) - _numbers(allowed))
    )
    problems.extend(
        f"unsupported_amount:{a}" for a in sorted(_amounts(text) - _amounts(allowed))
    )
    problems.extend(
        f"unsupported_percentage:{p}"
        for p in sorted(_percentages(text) - _percentages(allowed))
    )

    allowed_dates = _dates(allowed)
    compact_allowed = {d.replace(" ", "") for d in allowed_dates}
    problems.extend(
        f"unsupported_date:{date}"
        for date in sorted(_dates(text))
        if date not in allowed_dates and date.replace(" ", "") not in compact_allowed
    )

    allowed_lower = allowed.lower()
    lowered = text.lower()
    problems.extend(
        f"unsupported_name:{noun}"
        for noun in sorted(_proper_nouns(text))
        if _canonical_noun(noun) not in allowed_lower
    )
    problems.extend(
        f"internal_identifier:{token}"
        for token in sorted(set(_SNAKE_RE.findall(text)))
        if token not in allowed_lower
    )
    problems.extend(
        f"internal_detail:{term}"
        for term in _INTERNAL_TERMS
        if term in lowered and term not in allowed_lower
    )

    unquoted = _QUOTED_RE.sub("", text)
    if unquoted.count("?") > 1:
        problems.append("multiple_questions")
    if len(_REQUEST_RE.findall(unquoted)) > 1:
        problems.append("multiple_requests")
    ask = _ask_sentence(unquoted).lower()
    if sum(1 for verb in _ACTION_VERBS if verb in ask) > 1:
        problems.append("multiple_actions_in_one_ask")
    if brief.cta == "none" and "?" in unquoted:
        problems.append("question_added_to_information_only_message")

    problems.extend(
        f"taboo_term:{phrase}"
        for phrase in (t.split("(")[0].strip().lower() for t in brief.taboo_terms)
        if phrase and phrase in lowered
    )

    if brief.audience == "customer":
        problems.extend(
            f"merchant_analytics_shown_to_customer:{term}"
            for term in _MERCHANT_ONLY_TERMS
            if term in lowered and term not in allowed_lower
        )

    # An offer the draft named must survive the rewording. Without this a rewrite
    # can quietly swap one catalog item for another whose words are all ordinary
    # English ("Free Consultation" for "Aligner Consultation @ ₹499"), which no
    # other check would catch. Scoped to what the draft claimed, because a
    # mid-thread reply is not obliged to restate the offer at all.
    if brief.offer_title:
        service_name = brief.offer_title.split("@")[0].strip().lower()
        drafted_it = service_name and service_name in brief.deterministic_body.lower()
        if drafted_it and service_name not in lowered:
            problems.append("offer_changed_or_dropped")

    # A suggested offer must not be described as already running: that is a claim
    # the merchant can check on their own listing and find false.
    if brief.offer_title and not brief.offer_is_live:
        for claim in _LIVE_CLAIMS:
            if claim in lowered:
                problems.append(f"suggested_offer_described_as_live:{claim}")
                break

    return tuple(problems)
