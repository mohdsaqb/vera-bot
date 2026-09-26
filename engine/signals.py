"""Signal extraction: find the single strongest verifiable fact to say.

Each extractor returns `Signal`s built strictly from values present in the four
contexts, carrying the dotted path they came from. `select_primary_signal`
returns the one fact that earns the message (the "why now"), and
`select_supporting_fact` returns at most one more (the merchant/customer anchor).
Two facts is the cap on purpose — dumping every available number is what makes
messages unreadable and, per the rubric, no more specific.

Strengths are on 0-1 and only ever compare facts *within* a decision, so their
absolute values matter less than their order:
  0.90-1.00  trigger-supplied fact, verified against the merchant's own data
  0.70-0.89  material change or gap computed from the merchant's own numbers
  0.50-0.69  state facts (unverified listing, no active offer, dormancy)
  0.30-0.49  category-level context (peer stat, trend, seasonal beat)
"""

from __future__ import annotations

import re
from typing import Any

from engine import fmt, policy
from engine.normalize import days_between
from engine.types import (
    NormalizedCategory,
    NormalizedCustomer,
    NormalizedMerchant,
    NormalizedTrigger,
    Signal,
)

# Merchant metrics that have both a 30-day count and a 7-day delta.
TRACKED_METRICS: tuple[str, ...] = ("views", "calls", "directions", "leads")

# Peer benchmark key for each merchant metric.
PEER_KEY_FOR_METRIC: dict[str, str] = {
    "views": "avg_views_30d",
    "calls": "avg_calls_30d",
    "directions": "avg_directions_30d",
}


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def is_material_delta(merchant: NormalizedMerchant, metric: str) -> bool:
    """True when a 7-day delta is big enough, on a base big enough, to mention.

    Guards against the two ways a percentage misleads: a small swing, and a
    large swing on a handful of events (+5% of 14 calls is under one call).
    """
    delta = merchant.delta(metric)
    if delta is None or abs(delta) < policy.MIN_MATERIAL_DELTA:
        return False
    base = merchant.metric(metric)
    floor = policy.MIN_METRIC_BASE.get(metric)
    return floor is None or (base is not None and base >= floor)


def performance_delta_signal(
    merchant: NormalizedMerchant, metric: str, *, want: str = "any"
) -> Signal | None:
    """The merchant's own week-on-week move on one metric.

    `want` filters by direction ("down", "up" or "any") so a dip trigger can ask
    only for declines and refuse to dress a rise up as a fall.
    """
    delta = merchant.delta(metric)
    if delta is None or not is_material_delta(merchant, metric):
        return None
    if want == "down" and delta >= 0:
        return None
    if want == "up" and delta <= 0:
        return None

    base = merchant.metric(metric)
    where = fmt.direction(delta)
    label = policy.METRIC_LABELS.get(metric, metric)
    text = f"{label} are {where} {fmt.pct(delta)} week-on-week"
    if base is not None:
        window = merchant.performance.get("window_days", 30)
        text += f" ({fmt.count(base)} in the last {window} days)"
    return Signal(
        kind="perf_delta",
        text=text,
        strength=0.70 + min(0.15, abs(delta) / 2),
        source=f"merchant.performance.delta_7d.{metric}_pct",
        data={"metric": metric, "delta_pct": delta, "value": base},
    )


def strongest_delta_signal(
    merchant: NormalizedMerchant, *, want: str = "any"
) -> Signal | None:
    """The largest material move across all tracked metrics, in a fixed order."""
    candidates = [
        signal
        for metric in TRACKED_METRICS
        if (signal := performance_delta_signal(merchant, metric, want=want))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (abs(s.data.get("delta_pct", 0.0)), s.data["metric"]))


def peer_gap_signal(
    merchant: NormalizedMerchant, category: NormalizedCategory, *, want: str = "any"
) -> Signal | None:
    """Where this merchant sits against the category benchmark.

    CTR is compared in percentage points, which is the honest unit for a gap
    between two rates; volume metrics are compared as a ratio.
    """
    ctr = merchant.metric("ctr")
    peer_ctr = category.peer("avg_ctr")
    if ctr is not None and peer_ctr:
        gap = ctr - peer_ctr
        if abs(gap) * 100 >= policy.MIN_CTR_GAP_PP and (
            want == "any"
            or (want == "below" and gap < 0)
            or (want == "above" and gap > 0)
        ):
            side = "above" if gap > 0 else "below"
            scope = str(category.peer_stats.get("scope", "")).replace("_", " ").strip()
            tail = f" ({scope})" if scope else ""
            text = (
                f"your click-through rate is {fmt.rate_pct(ctr)} against the "
                f"{fmt.rate_pct(peer_ctr)} {category.display_name.lower()} median{tail} — "
                f"{fmt.points(gap)} {side}"
            )
            return Signal(
                kind="peer_gap",
                text=text,
                strength=0.72 if side == "below" else 0.66,
                source="merchant.performance.ctr vs category.peer_stats.avg_ctr",
                data={"metric": "ctr", "value": ctr, "peer": peer_ctr, "gap": gap, "side": side},
            )

    for metric, peer_key in PEER_KEY_FOR_METRIC.items():
        value = merchant.metric(metric)
        peer = category.peer(peer_key)
        if value is None or not peer:
            continue
        ratio = value / peer
        if want == "below" and ratio >= 0.75:
            continue
        if want == "above" and ratio <= 1.25:
            continue
        if 0.75 <= ratio <= 1.25:
            continue
        side = "above" if ratio > 1 else "below"
        label = policy.METRIC_LABELS.get(metric, metric)
        text = (
            f"{fmt.count(value)} {label} in 30 days against a category median of "
            f"{fmt.count(peer)} — {fmt.pct(abs(ratio - 1))} {side} peers"
        )
        return Signal(
            kind="peer_gap",
            text=text,
            strength=0.60,
            source=f"merchant.performance.{metric} vs category.peer_stats.{peer_key}",
            data={"metric": metric, "value": value, "peer": peer, "side": side},
        )
    return None


# --------------------------------------------------------------------------- #
# Account / listing state
# --------------------------------------------------------------------------- #
def listing_state_signal(merchant: NormalizedMerchant) -> Signal | None:
    """An unverified listing or a stale posting cadence, when recorded."""
    if merchant.verified is False:
        return Signal(
            kind="listing_unverified",
            text="your Google listing is still unverified",
            strength=0.62,
            source="merchant.identity.verified",
            data={"verified": False},
        )
    stale_days = merchant.signal_values.get("stale_posts")
    if isinstance(stale_days, int):
        return Signal(
            kind="stale_posts",
            text=f"your last Google post went up {stale_days} days ago",
            strength=0.58,
            source="merchant.signals.stale_posts",
            data={"stale_days": stale_days},
        )
    return None


def subscription_signal(merchant: NormalizedMerchant) -> Signal | None:
    """Subscription state, when it is close enough to matter."""
    if merchant.subscription_status == "expired" and merchant.days_since_expiry is not None:
        return Signal(
            kind="subscription_expired",
            text=(
                f"your {merchant.plan or 'subscription'} plan lapsed "
                f"{merchant.days_since_expiry} days ago"
            ),
            strength=0.68,
            source="merchant.subscription.days_since_expiry",
            data={"days_since_expiry": merchant.days_since_expiry},
        )
    days = merchant.days_remaining
    if days is not None and merchant.subscription_status in {"active", "trial"} and days <= 30:
        label = "trial" if merchant.subscription_status == "trial" else (merchant.plan or "plan")
        return Signal(
            kind="subscription_ending",
            text=f"your {label} has {days} days left",
            strength=0.66,
            source="merchant.subscription.days_remaining",
            data={"days_remaining": days, "status": merchant.subscription_status},
        )
    return None


def offer_gap_signal(merchant: NormalizedMerchant) -> Signal | None:
    """No active offer on the listing — a concrete, checkable gap."""
    if merchant.active_offers:
        return None
    expired = merchant.expired_offers
    if expired:
        return Signal(
            kind="offers_lapsed",
            text=f"you have no live offer right now — {expired[0].title} has ended",
            strength=0.60,
            source="merchant.offers[].status",
            data={"last_offer": expired[0].title},
        )
    return Signal(
        kind="offers_absent",
        text="there is no live offer on your listing at the moment",
        strength=0.54,
        source="merchant.offers",
        data={},
    )


def review_theme_signal(merchant: NormalizedMerchant, *, sentiment: str = "neg") -> Signal | None:
    """The most-repeated review theme of a given sentiment, when recorded."""
    themes = [
        theme
        for theme in merchant.review_themes
        if theme.get("sentiment") == sentiment and isinstance(theme.get("occurrences_30d"), int)
    ]
    if not themes:
        return None
    theme = max(themes, key=lambda t: (t["occurrences_30d"], str(t.get("theme", ""))))
    label = str(theme.get("theme", "")).replace("_", " ")
    occurrences = theme["occurrences_30d"]
    text = f"{occurrences} reviews in the last 30 days mention {label}"
    quote = theme.get("common_quote")
    if isinstance(quote, str) and quote:
        text += f' — one said "{quote}"'
    return Signal(
        kind="review_theme",
        text=text,
        strength=0.66 if sentiment == "neg" else 0.58,
        source="merchant.review_themes",
        data={"theme": theme.get("theme"), "occurrences_30d": occurrences, "quote": quote},
    )


def cohort_for_digest_signal(
    merchant: NormalizedMerchant, primary: Signal
) -> Signal | None:
    """Tie a research item's patient segment to this merchant's own roster.

    Only fires when both halves are present in the data: the digest item names a
    `patient_segment` and the merchant's `customer_aggregate` carries a count for
    it. That is what turns a category-level finding into a fact about this
    practice, and without both halves nothing is claimed.
    """
    segment = primary.data.get("patient_segment")
    if not isinstance(segment, str) or not segment:
        return None
    mapping = policy.SEGMENT_TO_AGGREGATE_KEY.get(segment)
    if mapping is None:
        return None
    key, label = mapping
    value = merchant.aggregate(key)
    if not isinstance(value, int) or value <= 0:
        return None
    return Signal(
        kind="cohort_match",
        text=f"that lands on the {fmt.count(value)} {label} on your roster",
        strength=0.86,
        source=f"category.digest.patient_segment + merchant.customer_aggregate.{key}",
        data={"segment": segment, key: value},
    )


def local_relevance_signal(
    merchant: NormalizedMerchant, primary: Signal
) -> Signal | None:
    """Note when a knowledge item belongs to the merchant's own city.

    Category digests carry their origin ("IDA Delhi chapter calendar"), so when
    that names the merchant's city the item is their local chapter's, not a
    generic national one. Both halves come from the data; without a match nothing
    is claimed.
    """
    city = merchant.city.strip()
    if not city:
        return None
    haystack = f"{primary.citation or ''} {primary.data.get('title', '')}".lower()
    if city.lower() not in haystack:
        return None
    return Signal(
        kind="local_chapter",
        text=f"it is your own {city} chapter running it",
        strength=0.70,
        source="category.digest.source + merchant.identity.city",
        data={"city": city},
    )


def digest_action_signal(primary: Signal) -> Signal | None:
    """The digest item's own `actionable` line — supplied, practice-level advice."""
    actionable = primary.data.get("actionable")
    if not isinstance(actionable, str) or not actionable:
        return None
    return Signal(
        kind="digest_action",
        text=actionable[0].lower() + actionable[1:] if actionable[0].isupper() else actionable,
        strength=0.80,
        source="category.digest.actionable",
        data={"actionable": actionable},
    )


def customer_base_signal(merchant: NormalizedMerchant) -> Signal | None:
    """The most useful number from the merchant's customer aggregate.

    Phrased in the category's own terms where the dataset carries a
    vertical-specific metric — a gym hears about membership churn, a restaurant
    about delivery share — because absent trade vocabulary reads as not having
    used the category context. Falls back to the generic keys every category has.
    """
    aggregate = merchant.customer_aggregate

    for key, template, kind in policy.AGGREGATE_FACTS_BY_CATEGORY.get(
        merchant.category_slug, ()
    ):
        value = aggregate.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            continue
        shown = fmt.rate_pct(float(value)) if kind == "rate" else fmt.count(float(value))
        return Signal(
            kind="customer_base",
            text=template.format(value=shown),
            strength=0.62 if kind == "rate" else 0.60,
            source=f"merchant.customer_aggregate.{key}",
            data={key: value},
        )

    for key, noun in (
        ("total_active_members", "active members"),
        ("total_unique_ytd", "unique customers this year"),
    ):
        value = aggregate.get(key)
        if isinstance(value, int) and value > 0:
            return Signal(
                kind="customer_base",
                text=f"you have {fmt.count(value)} {noun}",
                strength=0.56,
                source=f"merchant.customer_aggregate.{key}",
                data={key: value},
            )
    return None


def engagement_gap_signal(
    merchant: NormalizedMerchant, trigger: NormalizedTrigger
) -> Signal | None:
    """How long since the merchant last replied, from the trigger or history."""
    days = trigger.payload.get("days_since_last_merchant_message")
    if isinstance(days, int):
        return Signal(
            kind="dormancy",
            text=f"it has been {days} days since your last message to me",
            strength=0.68,
            source="trigger.payload.days_since_last_merchant_message",
            data={"days": days},
        )
    recorded = merchant.signal_values.get("dormant_with_vera")
    if isinstance(recorded, int):
        return Signal(
            kind="dormancy",
            text=f"we have not spoken in {recorded} days",
            strength=0.60,
            source="merchant.signals.dormant_with_vera",
            data={"days": recorded},
        )
    if not merchant.has_talked_before:
        return Signal(
            kind="dormancy",
            text="we have not spoken yet",
            strength=0.42,
            source="merchant.conversation_history",
            data={},
        )
    return None


# --------------------------------------------------------------------------- #
# Category knowledge
# --------------------------------------------------------------------------- #
def digest_signal(
    category: NormalizedCategory, trigger: NormalizedTrigger
) -> Signal | None:
    """The digest item the trigger points at, resolved from the category pack.

    Research and compliance claims carry their citation — the case studies cap
    the score at 7 for an uncited claim, and an unresolvable id must produce no
    claim at all rather than a plausible-sounding one.
    """
    item_id = trigger.payload.get("top_item_id") or trigger.payload.get("digest_item_id")
    item = category.digest_item(item_id if isinstance(item_id, str) else None)
    if item is None:
        item_id = trigger.payload.get("alert_id")
        item = category.digest_item(item_id if isinstance(item_id, str) else None)
    if item is None:
        return None

    text = item.title
    if item.kind == "cde":
        # For a CDE listing the useful detail is when it runs and what it is
        # worth, not the first line of the blurb.
        when = fmt.day_label(item.extra.get("date"))
        credits = item.extra.get("credits")
        bits = []
        if when:
            bits.append(f"on {when}")
        if isinstance(credits, (int, float)):
            bits.append(f"worth {credits:g} CDE credits")
        if bits:
            text = f"{text} — {fmt.join_terms(bits, last='and')}"
    else:
        detail = fmt.first_sentence(item.summary)
        if detail:
            text = f"{text} — {detail}"
        trial_n = item.extra.get("trial_n")
        if isinstance(trial_n, int):
            text += f" (n={fmt.count(trial_n)})"
    return Signal(
        kind=f"digest_{item.kind or 'item'}",
        text=text,
        strength=0.92,
        source=f"category.digest[{item.id}]",
        citation=item.source or None,
        data={
            "digest_id": item.id,
            "kind": item.kind,
            "title": item.title,
            "actionable": item.actionable,
            **{k: v for k, v in item.extra.items() if isinstance(v, (int, float, str))},
        },
    )


def trend_signal(category: NormalizedCategory) -> Signal | None:
    """The strongest category search trend, which is city-agnostic supplied data."""
    trends = [t for t in category.trend_signals if isinstance(t.get("delta_yoy"), (int, float))]
    if not trends:
        return None
    top = max(trends, key=lambda t: (t["delta_yoy"], str(t.get("query", ""))))
    query = str(top.get("query", "")).strip()
    text = f'"{query}" searches are up {fmt.pct(float(top["delta_yoy"]))} year-on-year'
    segment = top.get("segment_age")
    if isinstance(segment, str) and segment and segment != "all":
        text += f" in the {segment.replace('_', ' ')} band"
    return Signal(
        kind="category_trend",
        text=text,
        strength=0.46,
        source="category.trend_signals",
        citation=None,
        data={"query": query, "delta_yoy": top["delta_yoy"]},
    )


def seasonal_signal(
    category: NormalizedCategory, trigger: NormalizedTrigger
) -> Signal | None:
    """A seasonal beat, matched to the season the trigger names when it names one."""
    if not category.seasonal_beats:
        return None
    season = str(trigger.payload.get("season", "")).lower()
    month_hint = season.split("_")[0][:3].title() if season else ""
    chosen = None
    if month_hint:
        chosen = next(
            (b for b in category.seasonal_beats if month_hint in str(b.get("month_range", ""))),
            None,
        )
    if chosen is None:
        return None
    note = str(chosen.get("note", "")).strip()
    if not note:
        return None
    return Signal(
        kind="seasonal_beat",
        text=f"{chosen.get('month_range')} in your category is {note}",
        strength=0.50,
        source="category.seasonal_beats",
        data={"month_range": chosen.get("month_range"), "note": note},
    )


# --------------------------------------------------------------------------- #
# Trigger payload facts
# --------------------------------------------------------------------------- #
def _slots(trigger: NormalizedTrigger) -> list[dict[str, Any]]:
    for key in ("available_slots", "next_session_options"):
        value = trigger.payload.get(key)
        if isinstance(value, list):
            slots = [s for s in value if isinstance(s, dict) and s.get("label")]
            if slots:
                return slots
    return []


def trigger_payload_signal(
    trigger: NormalizedTrigger,
    merchant: NormalizedMerchant,
    customer: NormalizedCustomer | None,
) -> Signal | None:
    """The fact the trigger itself supplies, per family.

    This is the "why now" whenever the trigger carries real content. Returns None
    for the generated placeholder payloads, which forces the caller to fall back
    to merchant-derived facts or to decline.
    """
    payload = trigger.payload
    family = policy.family_for(trigger.kind)
    if trigger.is_placeholder:
        return None

    if family == "performance":
        metric = payload.get("metric")
        delta = payload.get("delta_pct")
        if isinstance(metric, str) and isinstance(delta, (int, float)):
            baseline = payload.get("vs_baseline")
            observed = merchant.metric(metric)
            where = fmt.direction(delta)
            label = policy.METRIC_LABELS.get(metric, metric)
            window = policy.WINDOW_LABELS.get(str(payload.get("window", "7d")), "7 days")
            text = f"{label} are {where} {fmt.pct(float(delta))} over the last {window}"
            if isinstance(baseline, (int, float)):
                text += f" against a baseline of {fmt.count(float(baseline))}"
            driver = payload.get("likely_driver")
            if isinstance(driver, str) and driver:
                text += f"; the likely driver is your {driver.replace('_', ' ')}"
            return Signal(
                kind="perf_delta",
                text=text,
                strength=0.94,
                source="trigger.payload.delta_pct",
                data={
                    "metric": metric,
                    "delta_pct": float(delta),
                    "baseline": baseline,
                    "value": observed,
                    "is_expected_seasonal": bool(payload.get("is_expected_seasonal")),
                    "season_note": payload.get("season_note"),
                },
            )

    if family == "competition":
        name = payload.get("competitor_name")
        if isinstance(name, str) and name:
            bits = [f"{name} opened"]
            distance = payload.get("distance_km")
            if isinstance(distance, (int, float)):
                bits.append(f"{distance:g} km away")
            opened = fmt.day_label(payload.get("opened_date"))
            if opened:
                bits.append(f"on {opened}")
            text = " ".join(bits)
            their_offer = payload.get("their_offer")
            if isinstance(their_offer, str) and their_offer:
                text += f", leading with {their_offer}"
            return Signal(
                kind="competitor",
                text=text,
                strength=0.90,
                source="trigger.payload.competitor_name",
                data={"competitor": name, "their_offer": their_offer, "distance_km": distance},
            )

    if family == "milestone":
        metric, now_value = payload.get("metric"), payload.get("value_now")
        target = payload.get("milestone_value")
        if isinstance(metric, str) and isinstance(now_value, (int, float)):
            label = policy.METRIC_LABELS.get(metric, metric.replace("_", " "))
            text = f"you are at {fmt.count(float(now_value))} {label}"
            if isinstance(target, (int, float)):
                gap = float(target) - float(now_value)
                text += f", {fmt.count(gap)} short of {fmt.count(float(target))}"
            return Signal(
                kind="milestone",
                text=text,
                strength=0.88,
                source="trigger.payload.value_now",
                data={"metric": metric, "value_now": now_value, "milestone_value": target},
            )

    if family == "reputation":
        theme, occurrences = payload.get("theme"), payload.get("occurrences_30d")
        if isinstance(theme, str) and isinstance(occurrences, int):
            text = f"{occurrences} reviews this month flag {theme.replace('_', ' ')}"
            trend = payload.get("trend")
            if isinstance(trend, str) and trend:
                text += f" and the trend is {trend}"
            quote = payload.get("common_quote")
            if isinstance(quote, str) and quote:
                text += f' — "{quote}"'
            return Signal(
                kind="review_theme",
                text=text,
                strength=0.90,
                source="trigger.payload.theme",
                data={"theme": theme, "occurrences_30d": occurrences, "quote": quote},
            )

    if family == "occasion" and trigger.kind == "ipl_match_today":
        match = payload.get("match")
        if isinstance(match, str) and match:
            bits = [match]
            venue = payload.get("venue")
            if isinstance(venue, str) and venue:
                bits.append(f"at {venue}")
            when = fmt.clock(payload.get("match_time_iso", ""))
            bits.append(f"tonight at {when}" if when else "tonight")
            text = " ".join(bits)
            return Signal(
                kind="match_day",
                text=text,
                strength=0.92,
                source="trigger.payload.match",
                data={
                    "match": match,
                    "venue": venue,
                    "is_weeknight": payload.get("is_weeknight"),
                    "city": payload.get("city"),
                },
            )

    if family == "occasion":
        festival = payload.get("festival")
        if isinstance(festival, str) and festival:
            days_until = payload.get("days_until")
            date_label = fmt.day_label(payload.get("date"))
            text = f"{festival} falls on {date_label}" if date_label else festival
            if isinstance(days_until, int):
                text += f", {days_until} days out"
            return Signal(
                kind="festival",
                text=text,
                strength=0.86,
                source="trigger.payload.festival",
                data={
                    "festival": festival,
                    "days_until": days_until,
                    "date": payload.get("date"),
                    "category_relevance": payload.get("category_relevance"),
                },
            )

    if family == "market":
        trends = payload.get("trends")
        if isinstance(trends, list) and trends:
            readable = [
                str(t).replace("_demand_", " demand ").replace("_", " ")
                for t in trends[:3]
                if isinstance(t, str)
            ]
            season = str(payload.get("season", "")).replace("_", " ").strip()
            text = f"{season} demand shift in your category: {fmt.join_terms(readable)}"
            return Signal(
                kind="category_seasonal",
                text=text.strip(),
                strength=0.84,
                source="trigger.payload.trends",
                data={"season": payload.get("season"), "trends": trends},
            )

    if family == "compliance":
        molecule = payload.get("molecule")
        batches = payload.get("affected_batches")
        if isinstance(molecule, str) and molecule:
            text = f"a recall is out on {molecule}"
            if isinstance(batches, list) and batches:
                text += f" batches {fmt.join_terms([str(b) for b in batches])}"
            maker = payload.get("manufacturer")
            if isinstance(maker, str) and maker:
                text += f" from {maker}"
            return Signal(
                kind="supply_alert",
                text=text,
                strength=0.96,
                source="trigger.payload.molecule",
                data={"molecule": molecule, "batches": batches, "manufacturer": maker},
            )

    if family == "planning":
        topic = payload.get("intent_topic")
        if isinstance(topic, str) and topic:
            text = f"you asked about a {topic.replace('_', ' ')}"
            said = payload.get("merchant_last_message")
            if isinstance(said, str) and said:
                text += f' — your words: "{said.strip()}"'
            return Signal(
                kind="planning_intent",
                text=text,
                strength=0.95,
                source="trigger.payload.intent_topic",
                data={"intent_topic": topic, "merchant_last_message": said},
            )

    if family == "listing" and payload.get("verified") is False:
        uplift = payload.get("estimated_uplift_pct")
        text = "your Google listing is unverified"
        # The uplift is the number that makes this worth acting on; without it the
        # fact is a status with nothing quantified, which reads as generic advice.
        # The verification path is logistics and belongs in the effort note, so it
        # only joins the fact when there is no uplift figure to lead with.
        path = payload.get("verification_path")
        if isinstance(uplift, (int, float)) and uplift:
            text += f", and verifying it is worth about {fmt.pct(float(uplift))} more views"
        elif isinstance(path, str) and path:
            text += f"; verification is by {path.replace('_', ' ')}"
        return Signal(
            kind="listing_unverified",
            text=text,
            strength=0.86,
            source="trigger.payload.verified",
            data={"verified": False, "estimated_uplift_pct": uplift, "path": path},
        )

    if family == "account":
        days_remaining = payload.get("days_remaining")
        if isinstance(days_remaining, int):
            amount = payload.get("renewal_amount")
            plan_name = payload.get("plan") or merchant.plan or "plan"
            text = f"your {plan_name} renews in {days_remaining} days"
            if isinstance(amount, (int, float)):
                text += f" at {fmt.rupees(amount)}"
            return Signal(
                kind="renewal",
                text=text,
                strength=0.90,
                source="trigger.payload.days_remaining",
                data={"days_remaining": days_remaining, "renewal_amount": amount},
            )
        since_expiry = payload.get("days_since_expiry")
        if isinstance(since_expiry, int):
            text = f"your plan lapsed {since_expiry} days ago"
            dip = payload.get("perf_dip_pct")
            if isinstance(dip, (int, float)):
                text += f" and traffic is {fmt.direction(float(dip))} {fmt.pct(float(dip))} since"
            return Signal(
                kind="winback",
                text=text,
                strength=0.88,
                source="trigger.payload.days_since_expiry",
                data={
                    "days_since_expiry": since_expiry,
                    "perf_dip_pct": dip,
                    "lapsed_added": payload.get("lapsed_customers_added_since_expiry"),
                },
            )

    if family == "knowledge":
        credits = payload.get("credits")
        if isinstance(credits, (int, float)):
            return Signal(
                kind="cde",
                text=f"{credits:g} CDE credits on offer",
                strength=0.80,
                source="trigger.payload.credits",
                data={"credits": credits, "fee": payload.get("fee")},
            )

    # ---- customer-scoped families ----
    if customer is None:
        return None

    if family == "recall":
        service = payload.get("service_due")
        due = fmt.day_label(payload.get("due_date"))
        last = payload.get("last_service_date") or customer.last_visit
        gap = days_between(last, payload.get("due_date"))
        label = str(service).replace("_", " ") if isinstance(service, str) else "check-up"
        text = f"your {label} is due"
        if due:
            text += f" around {due}"
        if isinstance(gap, int) and gap > 0:
            text += f", {gap // 30} months on from your last visit" if gap >= 60 else ""
        return Signal(
            kind="recall_due",
            text=text,
            strength=0.93,
            source="trigger.payload.service_due",
            data={
                "service_due": service,
                "due_date": payload.get("due_date"),
                "last_service_date": last,
                "slots": _slots(trigger),
            },
        )

    if family == "refill":
        molecules = payload.get("molecule_list")
        if isinstance(molecules, list) and molecules:
            names = [str(m) for m in molecules]
            runs_out = fmt.day_label(payload.get("stock_runs_out_iso"))
            text = f"your {fmt.join_terms(names)} run out"
            if runs_out:
                text += f" on {runs_out}"
            return Signal(
                kind="refill_due",
                text=text,
                strength=0.95,
                source="trigger.payload.molecule_list",
                data={
                    "molecules": names,
                    "runs_out": payload.get("stock_runs_out_iso"),
                    "delivery_saved": bool(payload.get("delivery_address_saved")),
                },
            )

    if family == "winback":
        days = payload.get("days_since_last_visit")
        focus = payload.get("previous_focus")
        if isinstance(days, int):
            # Exact days, not "about N weeks". The payload carries the precise
            # figure and a hedged round number reads as a form letter; the point
            # of the fact is that we know when they were last in.
            text = f"it has been {days} days since your last visit"
            if isinstance(focus, str) and focus:
                text += f", when you were working on {focus.replace('_', ' ')}"
            return Signal(
                kind="lapse",
                text=text,
                strength=0.90,
                source="trigger.payload.days_since_last_visit",
                data={
                    "days": days,
                    "focus": focus,
                    "months_member": payload.get("previous_membership_months"),
                },
            )

    if family == "trial":
        trial_date = fmt.day_label(payload.get("trial_date"))
        slots = _slots(trigger)
        if trial_date or slots:
            text = (
                f"you tried a session on {trial_date}"
                if trial_date
                else "you came in for a trial"
            )
            return Signal(
                kind="trial_done",
                text=text,
                strength=0.88,
                source="trigger.payload.trial_date",
                data={"trial_date": payload.get("trial_date"), "slots": slots},
            )

    if family == "occasion_customer":
        wedding = payload.get("wedding_date")
        days_to = payload.get("days_to_wedding")
        window = payload.get("next_step_window_open")
        if isinstance(days_to, int):
            text = f"{days_to} days to your wedding"
            if isinstance(window, str) and window:
                text += f", which opens your {readable_token(window)} window"
            return Signal(
                kind="wedding_window",
                text=text,
                strength=0.91,
                source="trigger.payload.days_to_wedding",
                data={
                    "days_to_wedding": days_to,
                    "wedding_date": wedding,
                    "window": window,
                    "trial_completed": payload.get("trial_completed"),
                },
            )

    if family == "appointment":
        when = payload.get("appointment_iso") or payload.get("appointment_time")
        label = fmt.day_label(when)
        if label:
            return Signal(
                kind="appointment",
                text=f"your appointment on {label}",
                strength=0.90,
                source="trigger.payload.appointment_iso",
                data={"appointment": when, "slots": _slots(trigger)},
            )
    return None


# --------------------------------------------------------------------------- #
# Customer-derived facts
# --------------------------------------------------------------------------- #
_SERVICE_NOISE = re.compile(r"_x\d+(_months?)?$|^chronic_rx_|_session_\d+$|_x\d+_")


_DURATION_SUFFIX = re.compile(r"^(.*?)_(\d+)\s*day$")


def readable_token(token: str) -> str:
    """Make a snake_case payload token read as prose.

    "skin_prep_program_30day" -> "30-day skin prep program": the dataset appends
    the duration, English puts it first.
    """
    text = str(token).strip()
    match = _DURATION_SUFFIX.match(text)
    if match:
        return f"{match.group(2)}-day {match.group(1).replace('_', ' ')}"
    return text.replace("_", " ")


def clean_service_label(service: str) -> str:
    """Turn a stored service token into a readable service name.

    The dataset encodes repeats and prefixes into the token
    ("membership_x4", "chronic_rx_metformin", "root_canal_session_1"); the
    repeat count and the prescription prefix are bookkeeping, not what the
    customer would call the visit.
    """
    cleaned = _SERVICE_NOISE.sub(" ", service).strip("_ ")
    return cleaned.replace("_", " ").strip()


def visit_count_signal(customer: NormalizedCustomer) -> Signal | None:
    """How many times this customer has been in, when that is more than once.

    A single visit is not a relationship worth citing; several is.
    """
    visits = customer.visits_total
    if not isinstance(visits, int) or visits < 2:
        return None
    return Signal(
        kind="customer_visits",
        text=f"you have been in {visits} times with us",
        strength=0.58,
        source="customer.relationship.visits_total",
        data={"visits_total": visits},
    )


def customer_relationship_signal(
    customer: NormalizedCustomer, now: Any = None
) -> Signal | None:
    """The most useful fact about this customer's history with this merchant."""
    services = [s for s in customer.services_received if s]
    if services:
        readable = [label for s in services if (label := clean_service_label(s))]
        top = max(set(readable), key=lambda s: (readable.count(s), s))
        visits = customer.visits_total
        text = (
            f"you have been in {visits} times, most often for {top}"
            if isinstance(visits, int) and visits > 1
            else f"you came in for {top}"
        )
        if not readable:
            return None
        return Signal(
            kind="customer_history",
            text=text,
            strength=0.72,
            source="customer.relationship.services_received",
            data={"visits_total": visits, "top_service": top},
        )

    gap = days_between(customer.last_visit, now) if now else None
    if isinstance(gap, int) and gap > 0:
        return Signal(
            kind="customer_gap",
            text=f"it has been {gap} days since your last visit",
            strength=0.64,
            source="customer.relationship.last_visit",
            data={"days_since_visit": gap},
        )
    if customer.last_visit:
        label = fmt.day_label(customer.last_visit)
        if label:
            return Signal(
                kind="customer_gap",
                text=f"your last visit with us was {label}",
                strength=0.56,
                source="customer.relationship.last_visit",
                data={"last_visit": customer.last_visit},
            )
    return None


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
# Per family, the merchant-derived facts to try when the trigger payload carries
# nothing usable. Order is the preference order.
_FALLBACK_CHAIN: dict[str, tuple[str, ...]] = {
    "performance": ("delta", "peer_gap"),
    "listing": ("listing", "peer_gap"),
    # Dormancy is deliberately last. "It has been N days since your last message
    # to me" is a fact about Vera's own inbox, not about the merchant's business,
    # so it justifies the timing but anchors nothing the merchant can act on;
    # it still reaches the message as the supporting "why now".
    # The peer benchmark leads rather than the subscription state: a merchant who
    # has gone quiet is usually one we last spoke to about their lapsed plan, and
    # leading on the plan again restates a story they have already been told. The
    # peer gap is the fact they have not heard.
    "reengagement": ("peer_gap", "delta", "dormancy"),
    "curiosity": ("delta", "peer_gap", "review_pos", "customer_base"),
    "account": ("subscription", "delta"),
    "reputation": ("review_neg",),
    "milestone": (),
    "competition": (),
    "occasion": (),
    "market": ("seasonal", "trend"),
    "knowledge": (),
    "compliance": (),
    "planning": (),
}


def _fallback_signal(
    name: str,
    merchant: NormalizedMerchant,
    category: NormalizedCategory,
    trigger: NormalizedTrigger,
) -> Signal | None:
    if name == "delta":
        want = "down" if trigger.kind in {"perf_dip", "seasonal_perf_dip"} else (
            "up" if trigger.kind == "perf_spike" else "any"
        )
        return strongest_delta_signal(merchant, want=want)
    if name == "peer_gap":
        return peer_gap_signal(merchant, category)
    if name == "listing":
        return listing_state_signal(merchant)
    if name == "dormancy":
        return engagement_gap_signal(merchant, trigger)
    if name == "subscription":
        return subscription_signal(merchant)
    if name == "review_neg":
        return review_theme_signal(merchant, sentiment="neg")
    if name == "review_pos":
        return review_theme_signal(merchant, sentiment="pos")
    if name == "customer_base":
        return customer_base_signal(merchant)
    if name == "seasonal":
        return seasonal_signal(category, trigger)
    if name == "trend":
        return trend_signal(category)
    return None


def select_primary_signal(
    category: NormalizedCategory,
    merchant: NormalizedMerchant,
    trigger: NormalizedTrigger,
    customer: NormalizedCustomer | None = None,
    now: Any = None,
) -> Signal | None:
    """The single strongest fact that justifies speaking now.

    Order of preference:
      1. what the trigger itself supplies (it defines the "why now"),
      2. the digest item it points at, for knowledge/compliance families,
      3. a fallback fact derived from context,
      4. nothing — the caller then declines rather than padding.

    For a customer-scoped trigger the fallback is restricted to facts about that
    customer's own relationship with the business. Merchant analytics are
    internal: a customer must never be told the shop's click-through rate.
    """
    family = policy.family_for(trigger.kind)
    digest = digest_signal(category, trigger)

    if family == "compliance":
        # A recall notice is only actionable with the specifics — molecule and
        # batch numbers live in the payload, the citation on the digest item, so
        # the payload fact wins and borrows the item's source.
        payload_fact = trigger_payload_signal(trigger, merchant, customer)
        if payload_fact is not None and payload_fact.data.get("batches"):
            citation = payload_fact.citation or (digest.citation if digest else None)
            extra = {"actionable": digest.data.get("actionable")} if digest else {}
            return Signal(
                kind=payload_fact.kind,
                text=payload_fact.text,
                strength=payload_fact.strength,
                source=payload_fact.source,
                citation=citation,
                data={**payload_fact.data, **extra},
            )
        if digest is not None:
            return digest
        if payload_fact is not None:
            return payload_fact

    if family == "knowledge" and digest is not None:
        return digest

    if (signal := trigger_payload_signal(trigger, merchant, customer)) is not None:
        return signal

    if digest is not None:
        return digest

    if trigger.scope == "customer":
        if customer is None:
            return None
        signal = customer_relationship_signal(customer, now)
        if signal is None or signal.kind not in policy.CUSTOMER_FACT_KINDS:
            return None
        return signal

    for name in _FALLBACK_CHAIN.get(family, ("delta", "peer_gap")):
        if (signal := _fallback_signal(name, merchant, category, trigger)) is not None:
            return signal
    return None


def _repeats(candidate: str, primary: str) -> bool:
    """True when a candidate fact restates content the primary fact already has."""
    stop = {"your", "you", "have", "been", "for", "the", "and", "with", "most", "often", "in",
            "times", "us", "it", "has", "a", "on", "of", "to", "is", "are"}
    words = {w.strip(".,;:\"'()") for w in candidate.lower().split()} - stop
    target = primary.lower()
    return any(len(w) > 4 and w in target for w in words)


def select_supporting_fact(
    category: NormalizedCategory,
    merchant: NormalizedMerchant,
    trigger: NormalizedTrigger,
    customer: NormalizedCustomer | None,
    primary: Signal,
    now: Any = None,
) -> Signal | None:
    """At most one more fact, anchoring the message on this specific recipient.

    Never repeats the primary fact's kind, and for customer-facing messages
    prefers the customer's own history over anything about the merchant's account.
    """
    if customer is not None:
        candidate = customer_relationship_signal(customer, now)
        # Either no second fact, or one that restates the first. The visit count
        # is the fallback in both cases: it is the one relationship fact every
        # customer record carries, and the generated customers carry no service
        # history at all, so without it those messages lose their second fact
        # entirely and read as a bare reminder.
        if candidate is None or candidate.kind == primary.kind or _repeats(
            candidate.text, primary.text
        ):
            return visit_count_signal(customer)
        return candidate

    family = policy.family_for(trigger.kind)
    if family in {"knowledge", "compliance"}:
        # A knowledge item is anchored either on the cohort it affects or on the
        # step it asks for — both supplied with the item. Practice analytics are
        # unrelated to a clinical finding and would read as a non-sequitur.
        cohort = cohort_for_digest_signal(merchant, primary)
        if cohort is not None:
            return cohort
        local = local_relevance_signal(merchant, primary)
        if local is not None:
            return local
        # Compliance splits by who the item touches: a product recall is about
        # the customers who bought it, a rule change is about the practice, so
        # the practice-level to-do line is the right anchor there.
        candidates = (
            (customer_base_signal(merchant), digest_action_signal(primary))
            if trigger.kind == "supply_alert"
            else (digest_action_signal(primary), customer_base_signal(merchant))
        )
        if family == "compliance":
            return next((c for c in candidates if c is not None), None)
        return digest_action_signal(primary)

    order = policy.SUPPORTING_FACT_ORDER.get(
        family, policy.DEFAULT_SUPPORTING_FACT_ORDER
    )
    for name in order:
        candidate = _fallback_signal(name, merchant, category, trigger)
        if candidate is None or candidate.kind == primary.kind:
            continue
        if candidate.source == primary.source:
            continue
        return candidate
    return None
