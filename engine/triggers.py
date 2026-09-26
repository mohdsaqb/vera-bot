"""Trigger evaluation: should Vera speak now, and about which trigger?

`evaluate_trigger` answers two things at once — whether a trigger is eligible at
all (expiry, category meaning, whether the merchant's data backs its claim,
consent when a customer is involved), and how strongly it competes for the
merchant's attention. `rank_triggers` orders a queue deterministically.

The scoring is a weighted sum of five components, each normalised to 0-1; the
weights and their justification live in `policy.PRIORITY_WEIGHTS`.
"""

from __future__ import annotations

from typing import Any

from engine import policy
from engine.normalize import days_between, parse_iso
from engine.signals import is_material_delta, select_primary_signal
from engine.types import (
    NormalizedCategory,
    NormalizedCustomer,
    NormalizedMerchant,
    NormalizedTrigger,
    TriggerEvaluation,
)

# Deadline fields a payload may carry. The soonest one drives time pressure.
_DEADLINE_KEYS: tuple[str, ...] = (
    "deadline_iso",
    "stock_runs_out_iso",
    "match_time_iso",
    "due_date",
    "date",
)


def _ineligible(trigger: NormalizedTrigger, code: str, note: str) -> TriggerEvaluation:
    return TriggerEvaluation(
        trigger_id=trigger.id, eligible=False, priority=0.0, reason_code=code, note=note
    )


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def urgency_score(trigger: NormalizedTrigger) -> float:
    """Map the dataset's 1-5 urgency onto 0-1."""
    return (trigger.urgency - 1) / 4


def category_fit_score(trigger: NormalizedTrigger, merchant: NormalizedMerchant) -> float:
    """1.0 when the kind means something for this vertical, 0.0 when it cannot.

    Some payloads state their own relevance (the Diwali trigger lists
    `category_relevance`); that supplied data wins over the static table.
    """
    allowed = policy.CATEGORY_RESTRICTED_KINDS.get(trigger.kind)
    if allowed is not None and merchant.category_slug not in allowed:
        return 0.0

    declared = trigger.payload.get("category_relevance")
    if isinstance(declared, list) and declared:
        return 1.0 if merchant.category_slug in declared else 0.0
    return 1.0


def merchant_relevance_score(
    trigger: NormalizedTrigger,
    merchant: NormalizedMerchant,
    category: NormalizedCategory,
) -> tuple[float, str]:
    """Does the merchant's own state back this trigger's claim?

    Returns (score, note). A score of 0.0 means contradicted — the merchant data
    says the opposite of the trigger — and the caller treats that as a hard block
    rather than a low rank, because sending it would mean asserting something the
    merchant can check and find false.
    """
    kind, payload = trigger.kind, trigger.payload

    if kind in {"perf_dip", "seasonal_perf_dip", "perf_spike"}:
        want = "up" if kind == "perf_spike" else "down"
        metric = payload.get("metric") if isinstance(payload.get("metric"), str) else None
        metrics = (metric,) if metric else ("calls", "views", "directions", "leads")
        for name in metrics:
            delta = merchant.delta(name)
            if delta is None:
                continue
            if want == "down" and delta < 0 and is_material_delta(merchant, name):
                return 1.0, f"merchant {name} down {abs(delta):.0%}"
            if want == "up" and delta > 0 and is_material_delta(merchant, name):
                return 1.0, f"merchant {name} up {delta:.0%}"
        # The trigger payload may carry the delta even when the snapshot does not.
        if isinstance(payload.get("delta_pct"), (int, float)):
            delta = float(payload["delta_pct"])
            if (want == "down" and delta < 0) or (want == "up" and delta > 0):
                return 0.85, "delta supplied by trigger payload only"
        return 0.0, f"no material {want}ward move in the merchant snapshot"

    if kind == "gbp_unverified":
        if merchant.verified is True:
            return 0.0, "merchant listing is already verified"
        return 1.0, "listing unverified"

    if kind == "renewal_due":
        days = payload.get("days_remaining")
        if isinstance(days, int) or merchant.days_remaining is not None:
            return 1.0, "subscription window known"
        return 0.5, "renewal window not in snapshot"

    if kind == "winback_eligible":
        if merchant.subscription_status == "expired" or merchant.days_since_expiry:
            return 1.0, "subscription lapsed"
        return 0.3, "no lapse recorded on the merchant"

    if kind == "dormant_with_vera":
        if isinstance(payload.get("days_since_last_merchant_message"), int):
            return 1.0, "dormancy period supplied"
        if merchant.signal_values.get("dormant_with_vera"):
            return 0.9, "dormancy flagged on the merchant"
        if not merchant.conversation_history:
            return 0.6, "no conversation history at all"
        return 0.4, "dormancy not quantified"

    if kind == "active_planning_intent":
        if isinstance(payload.get("intent_topic"), str) and payload["intent_topic"]:
            return 1.0, "merchant asked for this in conversation"
        intents = [
            turn for turn in merchant.conversation_history
            if str(turn.get("engagement", "")).startswith("intent")
        ]
        return (0.8, "intent recorded in conversation history") if intents else (
            0.0, "no planning intent in payload or history"
        )

    if kind in {"research_digest", "cde_opportunity", "regulation_change", "supply_alert"}:
        item_id = (
            payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
        )
        if category.digest_item(item_id if isinstance(item_id, str) else None) is not None:
            return 1.0, "digest item resolved in the category pack"
        if kind == "supply_alert" and isinstance(payload.get("molecule"), str):
            return 1.0, "alert details supplied in payload"
        return 0.0, "referenced knowledge item is not in the category context"

    if kind == "competitor_opened":
        if isinstance(payload.get("competitor_name"), str) and payload["competitor_name"]:
            return 1.0, "competitor named in payload"
        return 0.0, "competitor not named — would have to be invented"

    if kind == "milestone_reached":
        if isinstance(payload.get("value_now"), (int, float)):
            return 1.0, "milestone value supplied"
        return 0.0, "milestone value absent from payload and snapshot"

    if kind == "review_theme_emerged":
        if isinstance(payload.get("theme"), str) and payload["theme"]:
            return 1.0, "review theme supplied"
        return (1.0, "review theme in merchant snapshot") if merchant.review_themes else (
            0.0, "no review themes recorded"
        )

    if kind == "ipl_match_today":
        if not isinstance(payload.get("match"), str) or not payload["match"]:
            return 0.0, "match not named in payload"
        city = payload.get("city")
        elsewhere = (
            isinstance(city, str) and city and merchant.city
            and city.lower() != merchant.city.lower()
        )
        if elsewhere:
            return 0.2, f"match is in {city}, merchant is in {merchant.city}"
        return 1.0, "match details supplied for the merchant's city"

    if kind == "festival_upcoming":
        return (1.0, "festival named") if isinstance(payload.get("festival"), str) and payload[
            "festival"
        ] else (0.0, "festival not named — would have to be invented")

    if kind == "category_seasonal":
        if isinstance(payload.get("trends"), list) and payload["trends"]:
            return 1.0, "seasonal demand data supplied"
        return 0.4, "season named without demand detail"

    if kind == "curious_ask_due":
        # The curiosity family needs no payload facts — it asks the merchant a
        # question. It only needs a merchant worth asking and a live channel.
        return 1.0, "curious ask needs no payload facts"

    # Customer-scoped families: the customer record carries the relevance.
    if trigger.scope == "customer":
        return 0.8, "customer-scoped trigger"

    return 0.6, "no specific merchant precondition for this kind"


def time_pressure_score(trigger: NormalizedTrigger, now: Any = None) -> tuple[float, str]:
    """How much the closing window justifies interrupting now.

    Without a `now` from the caller there is no defensible way to compute
    elapsed time, so this returns a neutral score rather than guessing — the
    challenge is explicit that the current clock must not be assumed.
    """
    if now is None:
        return 0.4, "no reference time supplied"

    soonest: int | None = None
    label = ""
    for key in _DEADLINE_KEYS:
        gap = days_between(now, trigger.payload.get(key))
        if gap is not None and (soonest is None or gap < soonest):
            soonest, label = gap, key
    expiry_gap = days_between(now, trigger.expires_at)
    if expiry_gap is not None and (soonest is None or expiry_gap < soonest):
        soonest, label = expiry_gap, "expires_at"

    if soonest is None:
        return 0.4, "no dated field to measure against"
    if soonest <= 1:
        return 1.0, f"{label} is today or tomorrow"
    if soonest <= 7:
        return 0.8, f"{label} in {soonest} days"
    if soonest <= policy.NEAR_DEADLINE_DAYS:
        return 0.6, f"{label} in {soonest} days"
    return 0.3, f"{label} is {soonest} days out"


def _event_is_too_distant(trigger: NormalizedTrigger) -> tuple[bool, str]:
    """True for a dated event too far away to prepare for right now.

    A payload that names an open preparation window (the bridal trigger's
    `next_step_window_open`) is actionable however far the event is — that field
    is the dataset saying "the window is now".
    """
    if trigger.payload.get("next_step_window_open"):
        return False, ""
    for key in ("days_until", "days_to_wedding"):
        value = trigger.payload.get(key)
        if isinstance(value, int) and value > policy.MAX_EVENT_LEAD_DAYS:
            return True, f"{key}={value} is beyond the {policy.MAX_EVENT_LEAD_DAYS}-day lead window"
    return False, ""


def customer_state_check(
    trigger: NormalizedTrigger, customer: NormalizedCustomer | None
) -> tuple[bool, str]:
    """Does the customer's recorded state agree with what the trigger claims?

    `customer.state` is the dataset's own read on the relationship. A winback
    aimed at someone marked "active", or a trial follow-up for a long-standing
    member, is the trigger being wrong about the person — and acting on it would
    mean telling a customer something their own record contradicts.
    """
    if trigger.scope != "customer" or customer is None or not customer.state:
        return True, ""
    family = policy.family_for(trigger.kind)
    allowed = policy.STATES_FOR_FAMILY.get(family)
    if allowed is None or customer.state in allowed:
        return True, ""
    return False, (
        f"customer is recorded as {customer.state}, which does not match a "
        f"{trigger.kind} message"
    )


def consent_check(
    trigger: NormalizedTrigger, customer: NormalizedCustomer | None
) -> tuple[bool, str, str]:
    """Is customer-facing outreach permitted for this trigger's purpose?

    Returns (allowed, reason_code, note). Three distinct failures, because they
    are different problems: no reachable channel, an explicit opt-out, and
    consent that exists but not for this purpose.
    """
    if trigger.scope != "customer":
        return True, "", ""
    if customer is None:
        return False, policy.REASON_CUSTOMER_MISSING, "customer-scoped trigger, no customer context"
    if not customer.contactable:
        return False, policy.REASON_NOT_CONTACTABLE, "no recorded channel or phone for the customer"
    if customer.reminder_opt_in is False:
        return False, policy.REASON_OPTED_OUT, "customer has reminders switched off"
    if not customer.has_any_consent:
        return False, policy.REASON_NO_CONSENT, "no consent recorded at all"

    family = policy.family_for(trigger.kind)
    allowed_scopes = policy.CONSENT_SCOPES_BY_FAMILY.get(family, ())
    held = set(customer.consent_scope)
    # Report the most specific match: the family lists its scopes in order of
    # specificity, so the first one the customer holds is the best description.
    matched = next((scope for scope in allowed_scopes if scope in held), None)
    if matched:
        return True, "", f"consent held for {matched}"

    if policy.BROAD_MARKETING_SCOPE in held and not policy.is_clinical_family(family):
        return True, "", "broad promotional consent covers this purpose"

    return (
        False,
        policy.REASON_NO_CONSENT,
        f"consent is {sorted(held)} but this message needs one of {list(allowed_scopes)}",
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_trigger(
    category: NormalizedCategory | None,
    merchant: NormalizedMerchant | None,
    trigger: NormalizedTrigger | None,
    customer: NormalizedCustomer | None = None,
    now: Any = None,
) -> TriggerEvaluation:
    """Decide whether this trigger earns a message, and how strongly.

    Hard blocks are applied before scoring, because each of them describes a
    message that must not exist at all rather than one that ranks poorly.
    """
    if trigger is None:
        return TriggerEvaluation(
            trigger_id="",
            eligible=False,
            priority=0.0,
            reason_code=policy.REASON_MISSING_CONTEXT,
            note="no trigger context",
        )
    if merchant is None or category is None:
        missing = "merchant" if merchant is None else "category"
        return _ineligible(trigger, policy.REASON_MISSING_CONTEXT, f"{missing} context not loaded")

    expiry = parse_iso(trigger.expires_at)
    reference = parse_iso(now)
    if expiry is not None and reference is not None and expiry < reference:
        return _ineligible(trigger, policy.REASON_EXPIRED, f"expired at {trigger.expires_at}")

    fit = category_fit_score(trigger, merchant)
    if fit == 0.0:
        return _ineligible(
            trigger,
            policy.REASON_CATEGORY_MISMATCH,
            f"{trigger.kind} does not apply to {merchant.category_slug}",
        )

    allowed, consent_code, consent_note = consent_check(trigger, customer)
    if not allowed:
        return _ineligible(trigger, consent_code, consent_note)

    state_ok, state_note = customer_state_check(trigger, customer)
    if not state_ok:
        return _ineligible(trigger, policy.REASON_CONTRADICTED, state_note)

    distant, distant_note = _event_is_too_distant(trigger)
    if distant:
        return _ineligible(trigger, policy.REASON_EVENT_DISTANT, distant_note)

    relevance, relevance_note = merchant_relevance_score(trigger, merchant, category)
    if relevance == 0.0:
        return _ineligible(trigger, policy.REASON_CONTRADICTED, relevance_note)

    primary = select_primary_signal(category, merchant, trigger, customer, now)
    if primary is None:
        return _ineligible(
            trigger,
            policy.REASON_NO_FACT,
            f"no verifiable fact available for {trigger.kind}",
        )

    family = policy.family_for(trigger.kind)
    if family in policy.FAMILIES_REQUIRING_PAYLOAD_FACT and not primary.source.startswith(
        ("trigger.payload", "category.digest")
    ):
        return _ineligible(
            trigger,
            policy.REASON_NO_FACT,
            f"{trigger.kind} needs the detail from its own payload; "
            f"{primary.source} describes something else",
        )

    pressure, pressure_note = time_pressure_score(trigger, now)

    # A concrete fact is what makes an ask worth making, so actionable value
    # tracks the strength of the fact the message will be built on.
    actionable = min(1.0, primary.strength)

    components = {
        "urgency": urgency_score(trigger),
        "merchant_relevance": relevance,
        "actionable_value": actionable,
        "category_fit": fit,
        "time_pressure": pressure,
    }
    priority = round(
        sum(policy.PRIORITY_WEIGHTS[name] * value for name, value in components.items()), 4
    )

    if priority < policy.MIN_PRIORITY_TO_SEND:
        return TriggerEvaluation(
            trigger_id=trigger.id,
            eligible=False,
            priority=priority,
            components=components,
            reason_code=policy.REASON_LOW_PRIORITY,
            note=f"priority {priority:.2f} below {policy.MIN_PRIORITY_TO_SEND:.2f}",
        )

    return TriggerEvaluation(
        trigger_id=trigger.id,
        eligible=True,
        priority=priority,
        components=components,
        reason_code="eligible",
        note="; ".join(part for part in (relevance_note, pressure_note) if part),
    )


def rank_triggers(
    evaluations: list[TriggerEvaluation],
    triggers: dict[str, NormalizedTrigger],
) -> list[TriggerEvaluation]:
    """Order eligible evaluations best-first, deterministically.

    Ties break on urgency and then on trigger id, so the same queue always
    produces the same order regardless of input ordering or dict iteration.
    """
    eligible = [e for e in evaluations if e.eligible]
    return sorted(
        eligible,
        key=lambda e: (
            -e.priority,
            -(triggers[e.trigger_id].urgency if e.trigger_id in triggers else 0),
            e.trigger_id,
        ),
    )
