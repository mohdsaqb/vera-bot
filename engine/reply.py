"""What to do about an inbound reply. Deterministic, like every other decision.

`decide_reply` maps (conversation state, classified intent) onto one of the three
actions the contract allows, `send`, `wait`, `end`, and, when sending, drafts
the body from facts the conversation already holds. A writer may afterwards
reword that draft, but it cannot change the action, the CTA or the facts.

The intent-by-intent policy, and why each one is what it is:

    accept / action_request   Move to the action. The named failure in the brief
                              (§9, pattern D) is a bot that answers a commitment
                              with another qualifying question, so this branch
                              never asks one.
    reject                    Close. Pushing the same ask again is what makes
                              merchants mute the channel.
    delay                     Back off, do not close. "Later" is not "no".
    question                  Answer from the facts on hand, or say plainly that
                              the answer is not here. Never guess.
    off_topic                 Decline the detour in one clause, return to the ask.
    auto_reply                Flag it once for the owner, then back off, then
                              close. Production Vera burns 2-3 turns here.
    hostile                   Close immediately and suppress the merchant.
    ambiguous                 One short clarification, then back off.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine import intent as intents
from engine import policy
from engine.types import ConversationState, GenerationBrief

# Backoffs, in seconds. The auto-reply values follow the replay scenario in
# api-call-examples.md §4.1: a few hours after the first canned reply, a day
# after the second.
WAIT_AFTER_AUTO_REPLY = 14_400        # 4 hours: the owner may pick the phone up
WAIT_AFTER_SECOND_AUTO_REPLY = 86_400  # 24 hours
WAIT_AFTER_DELAY_SOON = 14_400         # "later", "in a bit"
WAIT_AFTER_DELAY_TOMORROW = 86_400     # "tomorrow", "next week"
WAIT_AFTER_AMBIGUOUS = 3_600
WAIT_DEFAULT = 7_200

_TOMORROW_MARKERS = ("tomorrow", "next week", "next month", "kal", "agle")

# Words the reply must not use when answering a commitment: asking any of these
# is the intent-handoff failure, and `judge_simulator.py` greps for them.
QUALIFYING_PHRASES = ("would you", "do you", "can you tell", "what if", "how about")


@dataclass(frozen=True)
class ReplyDecision:
    """One reply decision: the action, the draft, and why."""

    action: str  # "send" | "wait" | "end"
    intent: str
    rationale: str
    body: str = ""
    cta: str = "none"
    wait_seconds: int = 0
    end_reason: str = ""
    facts_used: tuple[str, ...] = ()

    @property
    def is_send(self) -> bool:
        return self.action == "send"


def _recipient(state: ConversationState) -> str:
    """First name to use, when the conversation knows one."""
    brief = state.brief
    if brief and brief.recipient_name:
        return brief.recipient_name.split()[0].strip(",.")
    return ""


def _address(state: ConversationState) -> str:
    """Closing address for a reply, when a name is known.

    Replies in a live thread do not open with a name: the reference replies in
    the case studies do not, and re-introducing yourself mid-thread is a listed
    anti-pattern. A name is warmer at the end of a sign-off.
    """
    name = _recipient(state)
    return f" Thanks {name}." if name else ""


def _subject(state: ConversationState) -> str:
    """A short noun phrase for what was being proposed.

    Prefers the real offer title, then the conversation's own ask, then a neutral
    "it": never a description the conversation does not hold.
    """
    brief = state.brief
    if brief and brief.offer_title:
        return brief.offer_title
    if state.action_key in {"draft_content", "compliance_checklist", "book_reminder"}:
        return "the draft"
    if state.action_key == "verify_listing":
        return "the verification"
    if state.action_key == "pull_list":
        return "the list"
    if state.action_key == "renew":
        return "the renewal"
    return "it"


# --------------------------------------------------------------------------- #
# Branches
# --------------------------------------------------------------------------- #
def _on_accept(state: ConversationState) -> ReplyDecision:
    """Move to the action. No further qualification, no claim of completion.

    Nothing here says the work is done: there is no execution tool behind this
    bot, so the honest move is to name the next step and ask for the one
    confirmation that releases it.
    """
    subject = _subject(state)
    if state.action_key == "pull_list":
        next_step = f"I'll put {subject} together and share it here for you to check"
    elif state.action_key == "verify_listing":
        next_step = f"I'll start {subject} and come back with what Google needs from you"
    elif state.action_key == "renew":
        next_step = f"I'll set {subject} up and send it here to approve before anything charges"
    elif state.action_key in {"book_slot", "offer_booking"}:
        next_step = "I'll hold that slot and send the confirmation here"
    elif state.action_key == "confirm_dispatch":
        next_step = "I'll get it dispatched and send the confirmation here"
    elif state.action_key == "activate_offer":
        next_step = f"I'll set {subject} up and send it here to approve before it goes live"
    else:
        next_step = f"I'll draft {subject} and send it here for your approval"

    body = f"Right — {next_step}. Reply CONFIRM and I'll proceed."
    return ReplyDecision(
        action="send",
        intent=intents.ACCEPT,
        rationale=(
            "Merchant committed explicitly; switched from proposing to executing and "
            "asked for a single confirmation rather than another qualifying question"
        ),
        body=body,
        cta="binary_confirm_cancel",
    )


def _on_reject(state: ConversationState) -> ReplyDecision:
    """Close without another attempt at the same ask."""
    return ReplyDecision(
        action="end",
        intent=intents.REJECT,
        rationale=(
            "Explicit no; closing the thread rather than re-pitching the same action, "
            "and suppressing this story for the merchant"
        ),
        end_reason="declined",
        body=f"Understood — I'll leave it there.{_address(state)}",
    )


def _on_hostile() -> ReplyDecision:
    """Close immediately. Nothing further goes out on this thread."""
    return ReplyDecision(
        action="end",
        intent=intents.HOSTILE,
        rationale=(
            "Merchant asked to be left alone; ending the conversation and suppressing "
            "further outreach to this merchant"
        ),
        end_reason="opted_out",
    )


def _on_delay(message: str) -> ReplyDecision:
    """Back off for as long as they asked for. This is not a rejection."""
    lowered = message.lower()
    seconds = (
        WAIT_AFTER_DELAY_TOMORROW
        if any(marker in lowered for marker in _TOMORROW_MARKERS)
        else WAIT_AFTER_DELAY_SOON
    )
    return ReplyDecision(
        action="wait",
        intent=intents.DELAY,
        rationale=(
            f"Merchant asked for time, which is not a refusal; backing off "
            f"{seconds // 3600}h and keeping the thread open"
        ),
        wait_seconds=seconds,
    )


def _on_question(state: ConversationState) -> ReplyDecision:
    """Answer from the facts this conversation holds, or say they are not here.

    The facts came from the decision that opened the thread, so restating one is
    grounded by construction. When none of them answers the question, the reply
    says so: a guess is the one thing that must not happen.
    """
    brief = state.brief
    facts = brief.facts if brief else ()
    ask = brief.ask if brief and brief.ask else ""

    if facts:
        answer = facts[0].rstrip(".")
        body = f"To be precise: {answer}.{' ' + ask if ask else ''}"
        return ReplyDecision(
            action="send",
            intent=intents.QUESTION,
            rationale=(
                "Answered from the fact the original decision was built on, then "
                "repeated the same single ask; nothing outside the stored context"
            ),
            body=body,
            cta=state.cta if state.cta != "none" else "open_ended",
            facts_used=(facts[0],),
        )

    return ReplyDecision(
        action="send",
        intent=intents.QUESTION,
        rationale=(
            "Question cannot be answered from the context this conversation holds; "
            "said so plainly instead of guessing"
        ),
        body=(
            "I don't have that detail on hand — I'd rather check than guess. "
            "Want me to come back with it?"
        ),
        cta="binary_yes_no",
    )


def _on_off_topic(state: ConversationState) -> ReplyDecision:
    """Decline the detour in one clause and return to the single ask."""
    brief = state.brief
    ask = brief.ask if brief and brief.ask else ""
    # With a live ask, return to it. Without one, invite something in scope: 
    # never "carry on with that", which would point back at the detour.
    tail = (
        f" {ask}" if ask
        else " Anything on your listing or offers I can pick up instead?"
    )
    return ReplyDecision(
        action="send",
        intent=intents.OFF_TOPIC,
        rationale=(
            "Out-of-scope request declined without pretending to help, then "
            "redirected to the original ask; thread kept open"
        ),
        body=f"That one's outside what I can help with.{tail}",
        cta=state.cta if state.cta != "none" else "open_ended",
    )


def _on_auto_reply(state: ConversationState, merchant_auto_replies: int) -> ReplyDecision:
    """Flag once, back off, then close.

    Counted across the merchant's threads, not just this one: the same number
    answers every conversation, so a merchant with an auto-responder switched on
    produces canned text wherever you write. Production Vera's biggest documented
    waste is spending two or three turns per thread here.
    """
    seen = max(state.consecutive_auto_replies, merchant_auto_replies)

    if seen <= 1:
        body = (
            "Looks like that came from an auto-responder. When you see this "
            "yourself, just reply YES and I'll pick it up."
        )
        return ReplyDecision(
            action="send",
            intent=intents.AUTO_REPLY,
            rationale=(
                "Detected a canned auto-reply; one explicit prompt for the owner "
                "rather than treating it as engagement"
            ),
            body=body,
            cta="binary_yes_no",
        )

    if seen == 2:
        return ReplyDecision(
            action="wait",
            intent=intents.AUTO_REPLY,
            rationale=(
                "Second canned reply in a row — the owner is not at the phone; "
                "backing off 24h instead of burning another turn"
            ),
            wait_seconds=WAIT_AFTER_SECOND_AUTO_REPLY,
        )

    return ReplyDecision(
        action="end",
        intent=intents.AUTO_REPLY,
        rationale=(
            f"{seen} canned replies and no human turn; zero engagement signal, "
            f"so closing rather than continuing to nudge"
        ),
        end_reason="auto_reply_only",
    )


def _on_ambiguous(state: ConversationState) -> ReplyDecision:
    """One short clarification, then back off, then close.

    A thread that has never produced a real reply runs out of patience sooner than
    one that has: "knowing when to stop" is a listed open challenge, and an
    unclear answer from someone who has engaged before is worth another turn in a
    way that one from a thread of canned text is not.
    """
    unclear_turns = sum(
        1 for turn in state.inbound if turn.intent == intents.AMBIGUOUS
    )
    patience = (
        policy.MAX_UNANSWERED_NUDGES if state.has_real_engagement
        else policy.MAX_UNANSWERED_NUDGES - 1
    )
    if state.unanswered_nudges >= patience or unclear_turns > patience:
        return ReplyDecision(
            action="end",
            intent=intents.AMBIGUOUS,
            rationale=(
                f"{state.unanswered_nudges} nudges and {unclear_turns} unclear "
                f"replies with"
                f"{'' if state.has_real_engagement else 'out'} any real engagement; "
                f"closing rather than continuing"
            ),
            end_reason="no_engagement",
        )

    already_clarified = any(
        turn.intent == intents.AMBIGUOUS for turn in state.inbound[:-1]
    )
    if already_clarified:
        return ReplyDecision(
            action="wait",
            intent=intents.AMBIGUOUS,
            rationale="Second unclear reply; backing off rather than guessing at intent",
            wait_seconds=WAIT_AFTER_AMBIGUOUS,
        )

    brief = state.brief
    ask = brief.ask if brief and brief.ask else "Want me to go ahead?"
    return ReplyDecision(
        action="send",
        intent=intents.AMBIGUOUS,
        rationale="Reply did not read either way; repeated the single ask plainly",
        body=f"Just so I don't guess — {ask[0].lower()}{ask[1:]}",
        cta=state.cta if state.cta != "none" else "binary_yes_no",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def decide_reply(
    state: ConversationState,
    message: str,
    result: intents.IntentResult,
    merchant_auto_replies: int = 0,
) -> ReplyDecision:
    """Decide what to do about one inbound message.

    Args:
        state: the thread, with the inbound turn already recorded.
        message: the raw inbound text.
        result: the deterministic classification of it.
        merchant_auto_replies: canned replies seen from this merchant across all
            threads.
    """
    if state.should_end and not state.active:
        return ReplyDecision(
            action="end",
            intent=result.intent,
            rationale=(
                f"Conversation already closed ({state.end_reason or 'ended'}); "
                f"nothing further will be sent on it"
            ),
            end_reason=state.end_reason or "already_closed",
        )

    branches = {
        intents.ACCEPT: lambda: _on_accept(state),
        intents.ACTION_REQUEST: lambda: _on_accept(state),
        intents.REJECT: lambda: _on_reject(state),
        intents.HOSTILE: _on_hostile,
        intents.DELAY: lambda: _on_delay(message),
        intents.QUESTION: lambda: _on_question(state),
        intents.OFF_TOPIC: lambda: _on_off_topic(state),
        intents.AUTO_REPLY: lambda: _on_auto_reply(state, merchant_auto_replies),
        intents.AMBIGUOUS: lambda: _on_ambiguous(state),
    }
    decision = branches.get(result.intent, lambda: _on_ambiguous(state))()

    # An acceptance is the one place where a stray qualifying question would undo
    # the whole transition, so it is checked rather than assumed, and repaired
    # in place rather than raised, because a request path must not fail here.
    if (
        decision.is_send
        and decision.intent in {intents.ACCEPT, intents.ACTION_REQUEST}
        and re_qualifies(decision.body)
    ):
        return ReplyDecision(
            action="send",
            intent=decision.intent,
            rationale=decision.rationale + " (draft rewritten to remove a qualifier)",
            body="Right — I'll get that moving and send it here for your "
                 "approval. Reply CONFIRM and I'll proceed.",
            cta="binary_confirm_cancel",
        )
    return decision


def re_qualifies(body: str) -> bool:
    """True when a body asks another qualifying question instead of acting.

    `judge_simulator.py` greps replies for exactly these phrases, and the brief
    names re-qualification after a commitment as production Vera's live failure.
    """
    lowered = body.lower()
    return any(phrase in lowered for phrase in QUALIFYING_PHRASES)


def brief_for_reply(
    state: ConversationState, decision: ReplyDecision, message: str
) -> GenerationBrief:
    """Project a reply decision into a brief a writer may reword.

    The facts allowed are the ones this conversation was built on, so a reply can
    restate them but cannot introduce anything new.
    """
    source = state.brief
    facts = decision.facts_used or (source.facts if source else ())
    return GenerationBrief(
        purpose="reply",
        category=source.category if source else state.category_slug,
        audience=state.audience,
        recipient_name=_recipient(state),
        salutation=_address(state).rstrip("— ").strip(),
        tone=source.tone if source else "",
        language=state.language,
        reason=f"reply to a {decision.intent} message",
        facts=facts,
        offer_title=source.offer_title if source else "",
        offer_is_live=source.offer_is_live if source else False,
        ask=decision.body,
        cta=decision.cta,
        citation="",
        taboo_terms=source.taboo_terms if source else (),
        style_notes=(
            source.style_notes if source else ()
        ) + (
            "This is a reply in a thread already underway — do not reintroduce "
            "yourself and do not repeat what was already said.",
            "Keep it to one or two short sentences.",
        ),
        deterministic_body=decision.body,
        inbound_message=message,
        intent=decision.intent,
    )
