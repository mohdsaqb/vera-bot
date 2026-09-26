"""Decision policy: the tables and thresholds every stage reads.

Keeping the tunable rules in one module means the engine's behaviour can be
argued about (and adjusted) in one place instead of being spread through the
pipeline. Each table below records *why* it holds the values it does, sourced
from the challenge documents and from the shape of the real dataset.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Trigger families
# --------------------------------------------------------------------------- #
# The dataset uses 24 trigger kinds. They group into families that share a
# message shape, an action and a consent purpose, which is what keeps the
# engine from needing 24 near-identical branches.
FAMILY_BY_KIND: Final[dict[str, str]] = {
    # merchant-facing, external knowledge
    "research_digest": "knowledge",
    "cde_opportunity": "knowledge",
    "regulation_change": "compliance",
    "supply_alert": "compliance",
    # merchant-facing, market context
    "category_trend_movement": "market",
    "category_seasonal": "market",
    "festival_upcoming": "occasion",
    "ipl_match_today": "occasion",
    "local_news_event": "occasion",
    "weather_heatwave": "occasion",
    "competitor_opened": "competition",
    # merchant-facing, own performance
    "perf_dip": "performance",
    "perf_spike": "performance",
    "seasonal_perf_dip": "performance",
    "milestone_reached": "milestone",
    "review_theme_emerged": "reputation",
    # merchant-facing, account state
    "renewal_due": "account",
    "winback_eligible": "account",
    "gbp_unverified": "listing",
    "dormant_with_vera": "reengagement",
    "curious_ask_due": "curiosity",
    "active_planning_intent": "planning",
    # customer-facing
    "recall_due": "recall",
    "appointment_tomorrow": "appointment",
    "chronic_refill_due": "refill",
    "customer_lapsed_soft": "winback",
    "customer_lapsed_hard": "winback",
    "trial_followup": "trial",
    "wedding_package_followup": "occasion_customer",
}

DEFAULT_FAMILY: Final[str] = "general"

# --------------------------------------------------------------------------- #
# Category fit
# --------------------------------------------------------------------------- #
# The expanded dataset assigns generated triggers to merchants at random, so a
# pharmacy-only kind can land on a dentist (test pair T08 is exactly that).
# Vertical-restricted kinds are listed here; anything absent is category-neutral.
# Restrictions come from what the kind *means*, not from a preference:
# a chronic-prescription refill only exists where prescriptions are dispensed.
CATEGORY_RESTRICTED_KINDS: Final[dict[str, frozenset[str]]] = {
    "chronic_refill_due": frozenset({"pharmacies"}),
    "supply_alert": frozenset({"pharmacies"}),
    "recall_due": frozenset({"dentists"}),
    "ipl_match_today": frozenset({"restaurants"}),
    "wedding_package_followup": frozenset({"salons"}),
    "cde_opportunity": frozenset({"dentists", "pharmacies"}),
    "regulation_change": frozenset({"dentists", "pharmacies", "restaurants"}),
}

# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #
# Customer-facing sends need consent *for that purpose*. The seed customers make
# the intent explicit: Priya's recall trigger pairs with "recall_reminders",
# Mr. Sharma's refill with "refill_reminders", Rashmi's winback with
# "winback_offers". Each family below lists the scopes that authorise it; a
# customer holding any one of them may be messaged for that family.
CONSENT_SCOPES_BY_FAMILY: Final[dict[str, tuple[str, ...]]] = {
    "recall": ("recall_reminders", "appointment_reminders", "treatment_followup"),
    "appointment": ("appointment_reminders",),
    "refill": ("refill_reminders", "delivery_notifications"),
    "winback": ("winback_offers", "promotional_offers"),
    "trial": ("kids_program_updates", "program_updates", "appointment_reminders"),
    "occasion_customer": ("bridal_package_followup", "appointment_reminders"),
    "compliance": ("recall_alerts", "refill_reminders"),
    "market": ("promotional_offers", "seasonal_health_content", "health_content"),
    "occasion": ("promotional_offers", "match_night_specials", "lunch_thali_updates"),
}

# Broad marketing consent. It authorises promotional outreach (winback, offers,
# seasonal) but never a clinical or medication-related reminder — sending a
# prescription refill note to someone who only agreed to receive offers is the
# kind of consent stretch the pharmacy and dentist voice rules rule out.
BROAD_MARKETING_SCOPE: Final[str] = "promotional_offers"
CLINICAL_FAMILIES: Final[frozenset[str]] = frozenset({"recall", "refill", "compliance"})

# --------------------------------------------------------------------------- #
# Trigger priority weights
# --------------------------------------------------------------------------- #
# Weights, not arbitrary constants — each maps to something the challenge scores
# or states:
#   urgency            the dataset sets 1-5 deliberately (5 = drug recall,
#                      1 = curious ask) and the design doc says triggers "rank
#                      against other queued triggers" by urgency. Largest share.
#   merchant_relevance a trigger whose claim the merchant's own data supports is
#                      what makes a message verifiable ("merchant fit" + the
#                      no-fabrication floor).
#   actionable_value   "engagement compulsion" needs a concrete next step.
#   category_fit       "category fit" is a scored dimension.
#   time_pressure      "why now" — a closing window earns interruption.
PRIORITY_WEIGHTS: Final[dict[str, float]] = {
    "urgency": 0.35,
    "merchant_relevance": 0.25,
    "actionable_value": 0.15,
    "category_fit": 0.15,
    "time_pressure": 0.10,
}

# Below this a trigger is not worth interrupting for, even when nothing is wrong
# with it. Calibrated so that a lowest-urgency trigger with no supporting
# merchant fact and no action (0.0 + 0.0 + 0.0 + fit + neutral pressure) falls
# under, while a low-urgency trigger that *does* have a fact and an action
# clears it (e.g. the curious-ask and milestone families).
MIN_PRIORITY_TO_SEND: Final[float] = 0.40

# --------------------------------------------------------------------------- #
# Materiality thresholds
# --------------------------------------------------------------------------- #
# A change has to be big enough to be worth a message, and big enough that the
# merchant will recognise it. 15% week-on-week is the floor the dataset's own
# perf triggers sit well above (-50%, -30%, +15%).
MIN_MATERIAL_DELTA: Final[float] = 0.15
# Percentage swings on tiny counts are noise: +5% of 14 calls is under one call.
MIN_METRIC_BASE: Final[dict[str, float]] = {
    "views": 200.0,
    "calls": 8.0,
    "directions": 10.0,
    "leads": 5.0,
}
# CTR gaps are reported in percentage points; below this the gap is not a story.
MIN_CTR_GAP_PP: Final[float] = 0.5

# A dated event earns a message only once it is close enough to prepare for.
# Kavya's bridal trigger shows the alternative route: a payload that names an
# open preparation window ("skin_prep_program_30day") is actionable regardless
# of how far the event itself is.
MAX_EVENT_LEAD_DAYS: Final[int] = 45
# Deadline-bearing triggers (compliance, refills) get full time pressure inside
# this window and taper outside it.
NEAR_DEADLINE_DAYS: Final[int] = 30

# Lapse windows, used only to describe a gap the dataset already records.
LAPSE_SOFT_DAYS: Final[int] = 90

# --------------------------------------------------------------------------- #
# Offer preference
# --------------------------------------------------------------------------- #
# challenge-brief.md §3 and §11: service+price beats a bare discount for Indian
# merchants ("Haircut @ ₹99" over "Flat 30% OFF"), and a flat percentage is an
# explicit anti-pattern when something more concrete exists.
OFFER_TYPE_RANK: Final[dict[str, int]] = {
    "service_at_price": 5,
    "free_service": 4,
    "free_trial": 4,
    "free_addon": 3,
    "membership": 2,
    "bogo": 2,
    "percentage_discount": 1,
}

# Which catalog audience suits which recipient state.
AUDIENCE_FOR_CUSTOMER_STATE: Final[dict[str, tuple[str, ...]]] = {
    "new": ("new_user", "all"),
    "active": ("repeat_user", "all", "new_user"),
    "lapsed_soft": ("new_user", "repeat_user", "all"),
    "lapsed_hard": ("new_user", "repeat_user", "all"),
    "churned": ("new_user", "all"),
}

# Words that tie an offer to a trigger family or a customer's own history.
FAMILY_OFFER_HINTS: Final[dict[str, tuple[str, ...]]] = {
    "recall": ("cleaning", "checkup", "consultation", "scaling"),
    "refill": ("delivery", "senior", "health card", "subscription", "refill"),
    "winback": ("trial", "first month", "free", "haircut", "membership"),
    "appointment": ("consultation", "checkup", "trial"),
    "trial": ("trial", "first month", "membership", "combo"),
    "occasion_customer": ("bridal", "trial", "package"),
    "occasion": ("combo", "thali", "brunch", "match", "family"),
    "performance": ("free", "consultation", "trial", "haircut"),
    "listing": ("free", "consultation"),
    "reengagement": ("free", "trial", "haircut", "consultation"),
    "competition": ("free", "consultation", "analysis", "cleaning"),
    "market": ("combo", "care", "delivery", "check"),
    "milestone": ("free", "combo", "family"),
}

# --------------------------------------------------------------------------- #
# Actions and CTAs
# --------------------------------------------------------------------------- #
# The CTA taxonomy is fixed by the API examples: open_ended, binary_yes_no,
# binary_confirm_cancel, multi_choice_slot, none. One primary CTA per message
# (challenge-brief.md §5.3 and §11).
CTA_OPEN: Final[str] = "open_ended"
CTA_BINARY: Final[str] = "binary_yes_no"
CTA_CONFIRM: Final[str] = "binary_confirm_cancel"
CTA_SLOTS: Final[str] = "multi_choice_slot"
CTA_NONE: Final[str] = "none"

# Effort framing — "effort externalization" is compulsion lever #4. Only ever a
# statement about Vera's own work, never an invented merchant-side promise.
EFFORT_NOTES: Final[dict[str, str]] = {
    "draft_content": "I'll have the draft ready in a couple of minutes",
    "activate_offer": "takes me a minute to set up",
    "verify_listing": "5-minute job at your end",
    "pull_list": "I'll have the list ready shortly",
    "book_slot": "",
    "confirm_dispatch": "",
    "ask_question": "one-line answer is enough",
    "renew": "",
    "share_info": "",
}

# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #
# Hindi-English code-mix is preferred for Hindi-speaking recipients
# (challenge-brief.md §5.7; ignoring the preference is a listed anti-pattern).
# Regional mixes (ta/te/kn) are rendered in English rather than substituting
# Hindi, which would be the wrong language for that recipient.
HINDI_MARKERS: Final[frozenset[str]] = frozenset({"hi", "hi-en mix", "hindi", "hi-en"})

# --------------------------------------------------------------------------- #
# No-action reason codes
# --------------------------------------------------------------------------- #
REASON_SENT: Final[str] = "sent"
REASON_EXPIRED: Final[str] = "trigger_expired"
REASON_CATEGORY_MISMATCH: Final[str] = "trigger_not_relevant_to_category"
REASON_CONTRADICTED: Final[str] = "trigger_claim_not_supported_by_data"
REASON_NO_FACT: Final[str] = "no_verifiable_fact_available"
REASON_NO_CONSENT: Final[str] = "customer_consent_missing_for_purpose"
REASON_OPTED_OUT: Final[str] = "customer_opted_out_of_reminders"
REASON_NOT_CONTACTABLE: Final[str] = "customer_not_contactable"
REASON_MISSING_CONTEXT: Final[str] = "required_context_missing"
REASON_EVENT_DISTANT: Final[str] = "event_too_distant_to_act_on"
REASON_LOW_PRIORITY: Final[str] = "not_worth_interrupting"
REASON_SUPPRESSED: Final[str] = "already_sent_for_this_suppression_key"
REASON_CUSTOMER_MISSING: Final[str] = "customer_context_required_but_absent"


def family_for(kind: str) -> str:
    """Trigger family for a kind, falling back to a neutral family."""
    return FAMILY_BY_KIND.get(kind, DEFAULT_FAMILY)


def is_clinical_family(family: str) -> bool:
    """True when the family carries a medical/medication purpose."""
    return family in CLINICAL_FAMILIES


def prefers_hindi_mix(language_tags: tuple[str, ...] | str) -> bool:
    """True when the recipient's language preference includes Hindi."""
    tags = (language_tags,) if isinstance(language_tags, str) else language_tags
    return any(tag.strip().lower() in HINDI_MARKERS for tag in tags if tag)


# --------------------------------------------------------------------------- #
# Offer relevance by family
# --------------------------------------------------------------------------- #
# An offer only belongs in the message when it is part of the value being
# offered or the thing being asked about. Bolting one onto a compliance notice,
# a CDE invite or a question to the merchant dilutes the single ask and reads as
# a sales bolt-on, which the rubric penalises as promotional drift.
FAMILIES_WITHOUT_OFFER: Final[frozenset[str]] = frozenset(
    {"curiosity", "knowledge", "compliance", "planning", "listing", "account", "reputation"}
)

# --------------------------------------------------------------------------- #
# Readable metric labels
# --------------------------------------------------------------------------- #
# Trigger payloads name metrics in snake_case ("review_count"); messages need the
# plain-English noun.
METRIC_LABELS: Final[dict[str, str]] = {
    "review_count": "reviews",
    "views": "profile views",
    "calls": "calls",
    "directions": "direction requests",
    "leads": "leads",
    "ctr": "click-through rate",
}

# Window shorthands as they appear in payloads.
WINDOW_LABELS: Final[dict[str, str]] = {
    "7d": "7 days",
    "30d": "30 days",
    "1d": "day",
}

# --------------------------------------------------------------------------- #
# Digest patient-segment -> merchant cohort count
# --------------------------------------------------------------------------- #
# A research item that names the segment it applies to can be tied to this
# merchant's own roster, which is what turns a category fact into a merchant
# fact ("relevant to your high-risk adult patients"). Both halves must exist in
# the data — the segment on the digest item, the count on the merchant.
SEGMENT_TO_AGGREGATE_KEY: Final[dict[str, tuple[str, str]]] = {
    "high_risk_adults": ("high_risk_adult_count", "high-risk adults"),
    "chronic": ("chronic_rx_count", "repeat-prescription customers"),
    "seniors": ("senior_customer_count", "senior customers"),
}

# Families where an offer is only worth naming if it actually fits the moment.
# For a bride in her skin-prep window, an unrelated haircut price is noise; for a
# lapsed customer being invited back, any live offer is a reason to return.
FAMILIES_NEEDING_RELEVANT_OFFER: Final[frozenset[str]] = frozenset(
    {"recall", "refill", "appointment", "trial", "occasion_customer", "occasion",
     "market", "competition", "milestone"}
)

# --------------------------------------------------------------------------- #
# Customer-facing fact sources
# --------------------------------------------------------------------------- #
# A message sent from the merchant's number to their customer may only be built
# from facts about that customer's own relationship with the business. Merchant
# analytics (profile views, click-through rate, peer benchmarks) are internal and
# must never be exposed to a customer (§9 of the phase brief).
CUSTOMER_FACT_KINDS: Final[frozenset[str]] = frozenset(
    {
        "recall_due",
        "refill_due",
        "lapse",
        "trial_done",
        "wedding_window",
        "appointment",
        "customer_history",
        "customer_gap",
    }
)


# --------------------------------------------------------------------------- #
# Families that cannot be built from context alone
# --------------------------------------------------------------------------- #
# Some trigger kinds *are* their payload: an appointment reminder with no
# appointment time, a competitor alert with no competitor, a milestone with no
# number. For these the primary fact must come from the trigger payload, or
# there is nothing to say that would not be invented. The 75 generated triggers
# carry `{"placeholder": true}`, so this is the rule that keeps them quiet
# instead of dressing an unrelated merchant metric up as the reason for writing.
FAMILIES_REQUIRING_PAYLOAD_FACT: Final[frozenset[str]] = frozenset(
    {"appointment", "recall", "refill", "milestone", "competition", "occasion",
     "occasion_customer", "trial"}
)

# --------------------------------------------------------------------------- #
# Customer state agreement
# --------------------------------------------------------------------------- #
# `customer.state` is the dataset's own read on the relationship, so a trigger
# family only applies when the two agree — a winback aimed at an "active"
# customer, or a trial follow-up for someone who has been coming for a year, is
# the trigger being wrong about the person.
STATES_FOR_FAMILY: Final[dict[str, frozenset[str]]] = {
    "winback": frozenset({"lapsed_soft", "lapsed_hard", "churned"}),
    "recall": frozenset({"active", "lapsed_soft", "lapsed_hard"}),
    "refill": frozenset({"active", "lapsed_soft"}),
    "trial": frozenset({"new"}),
    "occasion_customer": frozenset({"new", "active"}),
}

# --------------------------------------------------------------------------- #
# Customer-facing asks per category
# --------------------------------------------------------------------------- #
# Deliberately service-neutral: each is something any business of that type can
# honour without the engine inventing a service, a price or a slot.
WINBACK_ASK_BY_CATEGORY: Final[dict[str, tuple[str, str]]] = {
    "dentists": (
        "Reply YES and we will hold an appointment slot for you — no commitment.",
        "YES reply kijiye, hum appointment slot rakh dete hain — koi commitment nahi.",
    ),
    "salons": (
        "Reply YES and we will hold a slot for you — no commitment.",
        "YES reply kijiye, hum slot rakh dete hain — koi commitment nahi.",
    ),
    "gyms": (
        "Reply YES and we will hold a trial spot for you — no commitment.",
        "YES reply kijiye, hum trial spot rakh dete hain — koi commitment nahi.",
    ),
    "restaurants": (
        "Reply YES if you would like us to send you what is new on the menu.",
        "YES reply kijiye to hum aapko naye menu ki jaankari bhej denge.",
    ),
    "pharmacies": (
        "Reply YES if you would like us to set a refill reminder for you.",
        "YES reply kijiye to hum aapke refill ka reminder set kar denge.",
    ),
}


# --------------------------------------------------------------------------- #
# Conversation limits
# --------------------------------------------------------------------------- #
# "Knowing when to stop" is listed as an open challenge: after this many nudges
# with no real engagement a thread is closed rather than nudged again.
MAX_UNANSWERED_NUDGES: Final[int] = 3


# --------------------------------------------------------------------------- #
# Category-specific customer-base facts
# --------------------------------------------------------------------------- #
# Each vertical measures its customer base differently, and the case studies are
# explicit that absent trade vocabulary reads as not having used the category
# context at all. These map the aggregate keys that actually appear in the dataset
# onto the phrasing that category uses, so a gym hears about churn and a
# restaurant about delivery share rather than both hearing "customers".
#
# Entries are (aggregate key, phrasing template, kind) where the template takes
# the formatted value. Ordered by how directly each implies an action, and only
# keys the dataset actually carries are listed — nothing here invents a metric.
AGGREGATE_FACTS_BY_CATEGORY: Final[dict[str, tuple[tuple[str, str, str], ...]]] = {
    "gyms": (
        ("monthly_churn_pct", "membership churn is running at {value} a month", "rate"),
        ("trial_to_paid_pct", "{value} of your trials convert to paid", "rate"),
        ("total_active_members", "you have {value} active members", "count"),
    ),
    "restaurants": (
        ("delivery_share_pct", "delivery is {value} of your orders", "rate"),
        ("delivery_orders_30d", "{value} delivery orders in the last 30 days", "count"),
        ("dine_in_orders_30d", "{value} dine-in covers in the last 30 days", "count"),
        ("repeat_customer_pct", "{value} of your covers are repeat", "rate"),
    ),
    "salons": (
        ("retention_3mo_pct", "3-month retention is {value}", "rate"),
        ("lapsed_90d_plus", "{value} clients have not been in for 90 days", "count"),
        ("total_unique_ytd", "you have seen {value} clients this year", "count"),
    ),
    "pharmacies": (
        ("chronic_rx_count", "you have {value} customers on repeat prescriptions", "count"),
        ("repeat_customer_pct", "{value} of your counter is repeat business", "rate"),
    ),
    "dentists": (
        ("high_risk_adult_count", "{value} high-risk adults are on your roster", "count"),
        ("lapsed_180d_plus", "{value} patients have not been in for 180 days", "count"),
        ("retention_6mo_pct", "6-month recall retention is {value}", "rate"),
    ),
}


# --------------------------------------------------------------------------- #
# Supporting-fact preference, per family
# --------------------------------------------------------------------------- #
# Which second fact anchors the message best depends on what the message is
# about. Left to one global order, the peer click-through comparison wins almost
# every time, and the same benchmark sentence then appears in nearly every
# message across all five verticals — which is the absence of category voice the
# case studies penalise.
#
# So each family names its own preference, and the reasoning is the same each
# time: pick the fact a reader of *this* message would find relevant.
SUPPORTING_FACT_ORDER: Final[dict[str, tuple[str, ...]]] = {
    # The message is a metric diagnosis, so the benchmark explains it.
    "performance": ("peer_gap", "customer_base", "listing"),
    # You versus the market is the point.
    "competition": ("peer_gap", "customer_base", "delta"),
    # A milestone is about reputation; what people say beats what they click.
    "milestone": ("review_pos", "customer_base", "peer_gap"),
    # A timing play lands against the volume it would move.
    "occasion": ("customer_base", "delta", "peer_gap"),
    "market": ("customer_base", "peer_gap", "delta"),
    # Re-engagement is about the customers going quiet, not the click rate.
    "reengagement": ("customer_base", "peer_gap", "delta"),
    # A question opens best on whatever has just moved.
    "curiosity": ("delta", "customer_base", "peer_gap", "review_pos"),
    "listing": ("customer_base", "peer_gap"),
    "account": ("customer_base", "delta", "peer_gap"),
    "reputation": ("customer_base", "peer_gap"),
}

DEFAULT_SUPPORTING_FACT_ORDER: Final[tuple[str, ...]] = ("peer_gap", "delta", "customer_base")
