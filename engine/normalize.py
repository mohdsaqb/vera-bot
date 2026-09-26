"""Turn raw context JSON into the normalized dataclasses in `types.py`.

Only fields that actually exist in the challenge dataset are read: the field
names here were taken from `dataset/categories/*.json`,
`dataset/merchants_seed.json`, `dataset/customers_seed.json` and
`dataset/triggers_seed.json`, plus the generated variants written by
`dataset/generate_dataset.py` (which omit several optional blocks).

Every accessor is defensive: the judge may push a merchant with no `offers`,
no `signals` and no `review_themes` (all generated merchants look like that),
and a partial payload must never raise.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from engine.types import (
    DigestItem,
    NormalizedCategory,
    NormalizedCustomer,
    NormalizedMerchant,
    NormalizedTrigger,
    OfferTemplate,
    Voice,
)


# --------------------------------------------------------------------------- #
# Small readers
# --------------------------------------------------------------------------- #
def _dict(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def _list(payload: Any, key: str) -> list[Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return list(value) if isinstance(value, list) else []


def _str(payload: Any, key: str, default: str = "") -> str:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value.strip() if isinstance(value, str) else default


def _int(payload: Any, key: str) -> int | None:
    value = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _float(payload: Any, key: str) -> float | None:
    value = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _bool(payload: Any, key: str) -> bool | None:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, bool) else None


def _str_tuple(values: list[Any]) -> tuple[str, ...]:
    return tuple(v.strip() for v in values if isinstance(v, str) and v.strip())


def _dict_tuple(values: list[Any]) -> tuple[dict[str, Any], ...]:
    return tuple(v for v in values if isinstance(v, dict))


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp or date into an aware UTC datetime.

    Returns None for anything unparseable: timestamps arrive from the judge and
    from payloads in several shapes ("2026-11-12", "2026-04-28T00:00:00+05:30",
    "2026-05-03T00:00:00Z"), and an odd one must never break a decision.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def days_between(earlier: Any, later: Any) -> int | None:
    """Whole days from `earlier` to `later`, or None if either is unparseable."""
    start, end = parse_iso(earlier), parse_iso(later)
    if start is None or end is None:
        return None
    return (end - start).days


def infer_offer_type(title: str, declared: str = "") -> str:
    """Classify an offer from its title when the payload does not declare a type.

    Merchant offers in the dataset carry only id/title/status, while category
    catalog entries carry an explicit `type`. Inferring from the title lets the
    same preference ordering (a service at a named price beats a bare
    percentage) apply to both.
    """
    if declared:
        return declared
    text = title.lower()
    if "%" in text and "off" in text:
        return "percentage_discount"
    if "buy 1" in text or "bogo" in text:
        return "bogo"
    if "member" in text or "annual" in text or "health card" in text:
        return "membership"
    # "Free" wins over a rupee figure: in "Free Home Delivery > ₹499" the amount
    # is a qualifying threshold, not the price of the thing being offered.
    if "free" in text:
        conditional = any(word in text for word in (">", "with", "on orders"))
        return "free_addon" if conditional else "free_service"
    if "@" in text:
        return "service_at_price"
    return ""


def parse_signal_values(signals: tuple[str, ...]) -> dict[str, Any]:
    """Pull numbers out of the merchant's derived-signal strings.

    The dataset encodes them two ways, "stale_posts:22d" and
    "dormant_with_vera_14d", so both forms are decoded into
    `{"stale_posts": 22, "dormant_with_vera": 14}`. Flags without a number are
    recorded as True so callers can test presence uniformly.
    """
    values: dict[str, Any] = {}
    for signal in signals:
        name, _, suffix = signal.partition(":")
        if suffix:
            digits = "".join(ch for ch in suffix if ch.isdigit())
            values[name] = int(digits) if digits else suffix
            continue
        parts = signal.rsplit("_", 1)
        if len(parts) == 2 and parts[1][:-1].isdigit() and parts[1].endswith("d"):
            values[parts[0]] = int(parts[1][:-1])
        else:
            values[signal] = True
    return values


# --------------------------------------------------------------------------- #
# Category
# --------------------------------------------------------------------------- #
def normalize_category(payload: dict[str, Any] | None) -> NormalizedCategory | None:
    """Normalize a CategoryContext payload."""
    if not isinstance(payload, dict) or not payload:
        return None

    voice_raw = _dict(payload, "voice")
    voice = Voice(
        tone=_str(voice_raw, "tone"),
        register=_str(voice_raw, "register"),
        code_mix=_str(voice_raw, "code_mix"),
        vocab_allowed=_str_tuple(_list(voice_raw, "vocab_allowed")),
        vocab_taboo=_str_tuple(_list(voice_raw, "vocab_taboo")),
    )

    catalog = tuple(
        OfferTemplate(
            id=_str(item, "id"),
            title=_str(item, "title"),
            value=str(item.get("value", "")),
            audience=_str(item, "audience"),
            type=_str(item, "type"),
            source="category_catalog",
        )
        for item in _dict_tuple(_list(payload, "offer_catalog"))
        if _str(item, "title")
    )

    digest = tuple(
        DigestItem(
            id=_str(item, "id"),
            kind=_str(item, "kind"),
            title=_str(item, "title"),
            source=_str(item, "source"),
            summary=_str(item, "summary"),
            actionable=_str(item, "actionable"),
            extra={
                k: v
                for k, v in item.items()
                if k not in {"id", "kind", "title", "source", "summary", "actionable"}
            },
        )
        for item in _dict_tuple(_list(payload, "digest"))
        if _str(item, "title")
    )

    slug = _str(payload, "slug")
    return NormalizedCategory(
        slug=slug,
        display_name=_str(payload, "display_name", slug),
        voice=voice,
        offer_catalog=catalog,
        peer_stats=_dict(payload, "peer_stats"),
        digest=digest,
        content_library=_dict_tuple(_list(payload, "patient_content_library")),
        seasonal_beats=_dict_tuple(_list(payload, "seasonal_beats")),
        trend_signals=_dict_tuple(_list(payload, "trend_signals")),
        regulators=_str_tuple(_list(payload, "regulatory_authorities")),
        journals=_str_tuple(_list(payload, "professional_journals")),
        raw=payload,
    )


# --------------------------------------------------------------------------- #
# Merchant
# --------------------------------------------------------------------------- #
def normalize_merchant(payload: dict[str, Any] | None) -> NormalizedMerchant | None:
    """Normalize a MerchantContext payload."""
    if not isinstance(payload, dict) or not payload:
        return None

    identity = _dict(payload, "identity")
    subscription = _dict(payload, "subscription")
    performance = _dict(payload, "performance")
    signals = _str_tuple(_list(payload, "signals"))

    offers = tuple(
        OfferTemplate(
            id=_str(item, "id"),
            title=_str(item, "title"),
            value=str(item.get("value", "")),
            audience=_str(item, "audience"),
            type=infer_offer_type(_str(item, "title"), _str(item, "type")),
            status=_str(item, "status"),
            source="merchant_offers",
        )
        for item in _dict_tuple(_list(payload, "offers"))
        if _str(item, "title")
    )

    return NormalizedMerchant(
        merchant_id=_str(payload, "merchant_id"),
        category_slug=_str(payload, "category_slug"),
        name=_str(identity, "name"),
        owner_first_name=_str(identity, "owner_first_name"),
        city=_str(identity, "city"),
        locality=_str(identity, "locality"),
        languages=_str_tuple(_list(identity, "languages")),
        verified=_bool(identity, "verified"),
        established_year=_int(identity, "established_year"),
        subscription_status=_str(subscription, "status"),
        plan=_str(subscription, "plan"),
        days_remaining=_int(subscription, "days_remaining"),
        days_since_expiry=_int(subscription, "days_since_expiry"),
        performance=performance,
        delta_7d=_dict(performance, "delta_7d"),
        offers=offers,
        conversation_history=_dict_tuple(_list(payload, "conversation_history")),
        customer_aggregate=_dict(payload, "customer_aggregate"),
        signals=signals,
        signal_values=parse_signal_values(signals),
        review_themes=_dict_tuple(_list(payload, "review_themes")),
        raw=payload,
    )


# --------------------------------------------------------------------------- #
# Customer
# --------------------------------------------------------------------------- #
def normalize_customer(payload: dict[str, Any] | None) -> NormalizedCustomer | None:
    """Normalize a CustomerContext payload."""
    if not isinstance(payload, dict) or not payload:
        return None

    identity = _dict(payload, "identity")
    relationship = _dict(payload, "relationship")
    preferences = _dict(payload, "preferences")
    consent = _dict(payload, "consent")

    # A walk-in with no recorded phone and no channel cannot be reached at all
    # (c_015 in the seed data is exactly this case).
    channel = _str(preferences, "channel")
    has_phone = bool(_str(identity, "phone_redacted"))
    contactable = has_phone and channel not in {"", "none_recorded", "none"}

    reserved_rel = {
        "first_visit", "last_visit", "visits_total", "services_received", "lifetime_value",
    }
    reserved_pref = {"preferred_slots", "channel", "reminder_opt_in"}

    return NormalizedCustomer(
        customer_id=_str(payload, "customer_id"),
        merchant_id=_str(payload, "merchant_id"),
        name=_str(identity, "name"),
        language_pref=_str(identity, "language_pref"),
        age_band=_str(identity, "age_band"),
        is_senior=bool(identity.get("senior_citizen")),
        contactable=contactable,
        state=_str(payload, "state"),
        first_visit=_str(relationship, "first_visit"),
        last_visit=_str(relationship, "last_visit"),
        visits_total=_int(relationship, "visits_total"),
        services_received=_str_tuple(_list(relationship, "services_received")),
        lifetime_value=_float(relationship, "lifetime_value"),
        relationship_extra={k: v for k, v in relationship.items() if k not in reserved_rel},
        preferred_slots=_str(preferences, "preferred_slots"),
        channel=channel,
        reminder_opt_in=_bool(preferences, "reminder_opt_in"),
        preferences_extra={k: v for k, v in preferences.items() if k not in reserved_pref},
        opted_in_at=_str(consent, "opted_in_at"),
        consent_scope=_str_tuple(_list(consent, "scope")),
        raw=payload,
    )


# --------------------------------------------------------------------------- #
# Trigger
# --------------------------------------------------------------------------- #
def normalize_trigger(payload: dict[str, Any] | None) -> NormalizedTrigger | None:
    """Normalize a TriggerContext payload."""
    if not isinstance(payload, dict) or not payload:
        return None

    urgency = _int(payload, "urgency")
    customer_id = payload.get("customer_id")
    return NormalizedTrigger(
        id=_str(payload, "id"),
        kind=_str(payload, "kind"),
        scope=_str(payload, "scope", "merchant"),
        source=_str(payload, "source"),
        urgency=min(5, max(1, urgency)) if urgency is not None else 1,
        merchant_id=_str(payload, "merchant_id"),
        customer_id=customer_id if isinstance(customer_id, str) and customer_id else None,
        payload=_dict(payload, "payload"),
        suppression_key=_str(payload, "suppression_key"),
        expires_at=_str(payload, "expires_at"),
        raw=payload,
    )
