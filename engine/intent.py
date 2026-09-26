"""Deterministic classification of an inbound merchant or customer message.

No model is involved: intent decides what the bot does next, and that decision
stays in Python. Keyword and shape rules are applied in a fixed precedence,
because the same words mean different things depending on what else is present: "yes, but not now" is a delay, not an acceptance, and "thank you for contacting
us" is a machine, not a person.

Precedence, highest first:

    auto_reply      a canned business auto-response: a person did not write it
    hostile         abuse or an explicit demand to stop
    reject          a clear no
    delay           a yes-later
    off_topic       a subject outside what Vera does, however it is phrased
    question        they asked something
    action_request  an imperative: do the thing
    accept          a clear yes
    ambiguous       none of the above with enough confidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Intents
# --------------------------------------------------------------------------- #
ACCEPT = "accept"
ACTION_REQUEST = "action_request"
REJECT = "reject"
DELAY = "delay"
QUESTION = "question"
AUTO_REPLY = "auto_reply"
OFF_TOPIC = "off_topic"
HOSTILE = "hostile"
AMBIGUOUS = "ambiguous"

INTENTS: tuple[str, ...] = (
    ACCEPT, ACTION_REQUEST, REJECT, DELAY, QUESTION, AUTO_REPLY, OFF_TOPIC,
    HOSTILE, AMBIGUOUS,
)


@dataclass(frozen=True)
class IntentResult:
    """What the message was, and what in it said so."""

    intent: str
    confidence: float
    matched: str = ""
    note: str = ""


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
# Canned business auto-responses. These are phrases a WhatsApp Business
# auto-reply uses and a person answering a specific question would not.
_AUTO_REPLY_PATTERNS: tuple[str, ...] = (
    r"thank(?:s| you)[^.!?]{0,40}(?:for )?(?:contacting|reaching out|messaging|your message)",
    r"we (?:have )?received your (?:message|query|enquiry|inquiry)",
    r"(?:we|our team|someone) will (?:get back|respond|revert|reply|contact you)",
    r"our (?:team|representative|executive) will",
    r"(?:will|shall) (?:get back to you|revert) (?:shortly|soon|asap)",
    r"this is an automated (?:reply|response|message)",
    r"i am an automated assistant",
    r"main ek automated assistant hoon",
    r"aapki jaankari ke liye (?:bahut[- ]?)*(?:dhanyavaad|shukriya)",
    r"aapka (?:message|sandesh) (?:mil gaya|received)",
    r"(?:office|business) hours are",
    r"we are (?:currently )?(?:closed|unavailable|away)",
    r"out of office",
    r"kindly (?:wait|hold)[^.!?]{0,20}(?:we|our team)",
    r"hamari team[^.!?]{0,40}(?:pahuncha|sampark)",
)

# Abuse or an explicit demand to be left alone.
_HOSTILE_PATTERNS: tuple[str, ...] = (
    r"\bstop (?:messaging|texting|contacting|sending|spamming)",
    r"\b(?:do ?n[o']?t|never) (?:message|contact|text|call) me",
    r"\b(?:this is|you are|your service is) (?:useless|rubbish|garbage|nonsense|spam|pathetic)",
    r"\b(?:waste of|wasting) my time",
    r"\b(?:bakwas|bekar|bandh karo|pareshan)",
    r"\bunsubscribe\b",
    r"\bremove (?:me|my number)\b",
    r"\bwhy are you (?:bothering|disturbing|harassing)",
    r"\bleave me alone\b",
    r"\bshut up\b",
    r"\b(?:f[u\*]ck|bloody|idiot|stupid bot)\b",
)

# A clear no.
_REJECT_PATTERNS: tuple[str, ...] = (
    r"\bnot interested\b",
    r"\bno,? thank",
    r"\bno thanks\b",
    r"^\s*no[\s.!]*$",
    r"\bdo ?n[o']?t (?:want|need|bother|do)\b",
    r"\b(?:don'?t|do not) proceed\b",
    r"\bskip (?:it|this|that)\b",
    r"\bcancel (?:it|this|that)\b",
    r"\bnot (?:required|needed|for me|now or later)\b",
    r"\bno need\b",
    r"\bnahi chahiye\b",
    r"\bmat (?:karo|bhejo)\b",
    r"\brehne do\b",
    r"\bplease don'?t\b",
)

# A yes, but later.
_DELAY_PATTERNS: tuple[str, ...] = (
    r"\bnot (?:right )?now\b",
    r"\blater\b",
    r"\btomorrow\b",
    r"\bnext (?:week|month|time)\b",
    r"\bin (?:a|an|\d+) (?:few )?(?:hour|hours|day|days|week|weeks|min|minutes)\b",
    r"\b(?:i'?ll|i will|let me) (?:decide|think|check|get back|revert|see)\b",
    r"\bgive me (?:some )?time\b",
    r"\bremind me\b",
    r"\bbusy (?:right )?now\b",
    r"\bbaad mein\b",
    r"\bkal\b",
    r"\bthodi der\b",
    r"\babhi nahi\b",
    r"\bhold (?:on|off)\b",
)

# An imperative: do it.
_ACTION_REQUEST_PATTERNS: tuple[str, ...] = (
    r"\b(?:activate|publish|launch|schedule|book|send|post|set) (?:it|this|that|them|now)\b",
    r"\b(?:go ahead and|please) (?:activate|publish|send|post|book|draft|do)\b",
    r"\bdo (?:it|this|that) (?:now|today|asap)\b",
    r"^\s*(?:confirm|confirmed|proceed|go|start)[\s.!]*$",
    r"\bsend (?:me|it|the|them)\b",
    r"\bshare (?:it|the|them)\b",
    r"\bpull (?:the|it)\b",
    r"\bkar do\b",
    r"\bbhej do\b",
    r"\bshuru kar\b",
    r"\bactivate kar\b",
)

# A clear yes. Kept to leading affirmations and unmistakable commitments so a
# "yes" buried in a longer sentence cannot carry the whole message.
_ACCEPT_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:yes|yeah|yep|yup|ya|haan|ha|ok|okay|sure|fine|done|great|perfect)\b",
    r"\b(?:let'?s|lets) (?:do|go)\b",
    r"\bgo ahead\b",
    r"\bplease (?:do|proceed)\b",
    r"\bi'?m in\b",
    r"\bsounds (?:good|great|fine)\b",
    r"\bthat works\b",
    r"\bagreed\b",
    r"\bkar dijiye\b",
    r"\bkar dena\b",
    r"\btheek hai\b",
    r"\bchalega\b",
    r"\bbilkul\b",
    r"\byes,? (?:do|go|please|send|activate|proceed)\b",
)

# Question shapes, when no clearer intent applies.
_QUESTION_PATTERNS: tuple[str, ...] = (
    r"\bhow (?:much|many|long|will|does it)\b",
    r"\bwhat (?:is|are|do|does|kind|sort|exactly)\b",
    r"\bcan you (?:explain|tell|clarify|confirm)\b",
    r"\bwhat do you mean\b",
    r"\bkitna\b",
    r"\bkaise\b",
    r"\bkya (?:hai|hota|matlab)\b",
    r"\bmatlab\b",
)

# A leading interrogative counts only when the message is punctuated as a
# question: "do it now" is an instruction that happens to start with "do".
_LEADING_INTERROGATIVE_RE = re.compile(
    r"^\s*(?:how|what|why|when|where|which|who|can|could|do|does|is|are|will"
    r"|would|should)\b",
    re.IGNORECASE,
)

# Things Vera does not do. Named explicitly so the redirect is honest rather
# than a guess at what "off-topic" means.
_OFF_TOPIC_PATTERNS: tuple[str, ...] = (
    r"\bgst\b",
    r"\b(?:income )?tax (?:filing|return)\b",
    r"\bitr\b",
    r"\bloans?\b",
    r"\b(?:bank|banking) (?:account|details)\b",
    r"\b(?:rent|lease|landlord)s?\b",
    r"\b(?:hiring|recruit|staff salary|payroll)\b",
    r"\b(?:license|licence|fssai) (?:renewal|application)\b",
    r"\belectricity bills?\b",
    r"\binsurance\b",
    r"\blegal (?:notice|case)\b",
    r"\baccountant\b|\bca\b",
)

_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    AUTO_REPLY: tuple(re.compile(p, re.IGNORECASE) for p in _AUTO_REPLY_PATTERNS),
    HOSTILE: tuple(re.compile(p, re.IGNORECASE) for p in _HOSTILE_PATTERNS),
    REJECT: tuple(re.compile(p, re.IGNORECASE) for p in _REJECT_PATTERNS),
    DELAY: tuple(re.compile(p, re.IGNORECASE) for p in _DELAY_PATTERNS),
    QUESTION: tuple(re.compile(p, re.IGNORECASE) for p in _QUESTION_PATTERNS),
    ACTION_REQUEST: tuple(re.compile(p, re.IGNORECASE) for p in _ACTION_REQUEST_PATTERNS),
    ACCEPT: tuple(re.compile(p, re.IGNORECASE) for p in _ACCEPT_PATTERNS),
    OFF_TOPIC: tuple(re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS),
}

# Order matters: the first family that matches wins. Subject matter is placed
# above question shape: "can you do my GST?" needs redirecting, not answering.
_PRECEDENCE: tuple[str, ...] = (
    AUTO_REPLY, HOSTILE, REJECT, DELAY, OFF_TOPIC, QUESTION, ACTION_REQUEST, ACCEPT,
)

# A slot pick ("1", "2", "Wed") answers a multi-choice ask.
_SLOT_REPLY_RE = re.compile(r"^\s*(?:option\s*)?[12ab]\s*$", re.IGNORECASE)

# Questions that come *with* a commitment rather than instead of one. "Ok let's
# do it, what's next?" is an acceptance asking for the next step, and treating it
# as a plain question is the intent-handoff failure the brief calls out by name.
_FOLLOW_THROUGH_RE = re.compile(
    r"\b(?:what(?:'?s| is)? next|what next|then what|what do i (?:do|need)"
    r"|how (?:do|should) (?:we|i) (?:start|begin|proceed)|aage kya|ab kya)\b",
    re.IGNORECASE,
)


def normalise(message: str) -> str:
    """Collapse whitespace and strip surrounding punctuation for matching."""
    return " ".join((message or "").split()).strip()


def looks_like_auto_reply(message: str) -> bool:
    """True when the text reads as a canned business auto-response."""
    text = normalise(message)
    return any(pattern.search(text) for pattern in _COMPILED[AUTO_REPLY])


def classify_reply(
    message: str,
    previous_messages: tuple[str, ...] = (),
    expects_slot: bool = False,
) -> IntentResult:
    """Classify one inbound message.

    Args:
        message: the raw inbound text.
        previous_messages: what this counterpart sent before, newest last. A
            verbatim repeat is strong evidence of a machine even when the text
            itself carries no canned phrase.
        expects_slot: True when the last outbound asked the recipient to pick a
            slot, which makes a bare "1" an acceptance rather than noise.
    """
    text = normalise(message)
    if not text:
        return IntentResult(AMBIGUOUS, 0.0, note="empty message")

    # A verbatim repeat of an earlier message is machine behaviour.
    if previous_messages and text.lower() == normalise(previous_messages[-1]).lower():
        return IntentResult(
            AUTO_REPLY, 0.9, matched="verbatim repeat",
            note="identical to the previous inbound message",
        )

    if expects_slot and _SLOT_REPLY_RE.match(text):
        return IntentResult(ACCEPT, 0.9, matched=text, note="slot selection")

    is_question = text.endswith("?")
    follows_through = bool(_FOLLOW_THROUGH_RE.search(text))

    # "Ok, let's do it: what's next?" is a commitment that asks how to proceed,
    # not a question instead of a commitment. Checked ahead of precedence because
    # the follow-through half would otherwise classify as a plain question, which
    # is the intent-handoff failure the brief names.
    if follows_through and not any(
        pattern.search(text)
        for intent in (AUTO_REPLY, HOSTILE, REJECT, DELAY)
        for pattern in _COMPILED[intent]
    ):
        for intent in (ACCEPT, ACTION_REQUEST):
            for pattern in _COMPILED[intent]:
                match = pattern.search(text)
                if match:
                    return IntentResult(
                        ACCEPT, 0.85, matched=match.group(0),
                        note="commitment followed by a how-do-we-proceed question",
                    )

    for intent in _PRECEDENCE:
        for pattern in _COMPILED[intent]:
            match = pattern.search(text)
            if not match:
                continue
            # A question mark normally outranks a bare keyword: "activate it?"
            # is asking, not instructing: except when the question is asking how
            # to proceed, which is part of the commitment, not a substitute for
            # it. Rejection, hostility and subject matter are never softened.
            if is_question and not follows_through and intent in {ACTION_REQUEST, ACCEPT}:
                return IntentResult(
                    QUESTION, 0.6, matched=match.group(0),
                    note=f"{intent} wording but phrased as a question",
                )
            return IntentResult(intent, 0.85, matched=match.group(0))

    if is_question and _LEADING_INTERROGATIVE_RE.match(text):
        return IntentResult(QUESTION, 0.7, matched=text.split()[0], note="leading interrogative")
    # A question mark on its own is only a question when there is a question
    # under it; "k?" carries no intent at all.
    if is_question and len(text) >= 8:
        return IntentResult(QUESTION, 0.5, matched="?", note="question mark only")
    return IntentResult(AMBIGUOUS, 0.3, note="no recognised intent")
