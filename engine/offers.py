"""Offer selection: pick one real offer, or none.

Two sources, in strict preference order:

1. `merchant.offers` with `status == "active"`: something the merchant already
   runs. Referencing it costs the merchant nothing and is verifiable.
2. `category.offer_catalog`: the canonical service+price patterns supplied with
   the challenge. These are real supplied data, but they are *not* live for this
   merchant, so a choice from here is marked `is_existing=False` and the renderer
   must phrase it as a proposal ("want me to set it up?"), never as live.

Nothing is ever synthesised: no invented price, discount, expiry or service.
Expired merchant offers are never offered: they may only appear as background
in a signal (see `signals.offer_gap_signal`).
"""

from __future__ import annotations

from engine import policy
from engine.types import (
    NormalizedCategory,
    NormalizedCustomer,
    NormalizedMerchant,
    NormalizedTrigger,
    OfferChoice,
    OfferTemplate,
)


def _keyword_score(offer: OfferTemplate, hints: tuple[str, ...]) -> int:
    """How many family hints appear in the offer title."""
    title = offer.title.lower()
    return sum(1 for hint in hints if hint in title)


def _customer_affinity(offer: OfferTemplate, customer: NormalizedCustomer | None) -> int:
    """Bonus when the offer matches this customer's own history or status."""
    if customer is None:
        return 0
    title = offer.title.lower()
    score = 0
    for service in customer.services_received:
        token = service.replace("_", " ").split()[0] if service else ""
        if token and token in title:
            score += 2
    # Merchant offers carry no `audience` field, so a senior-specific offer is
    # recognised from its title as well as from the catalog's audience tag.
    if customer.is_senior and (offer.audience == "senior" or "senior" in title):
        score += 3
    return score


def _audience_rank(offer: OfferTemplate, customer: NormalizedCustomer | None) -> int:
    """Rank the offer's audience against the recipient, best first (higher wins)."""
    if customer is None:
        return 0
    preferred = policy.AUDIENCE_FOR_CUSTOMER_STATE.get(customer.state, ("all",))
    if not offer.audience:
        return 0
    if offer.audience in preferred:
        return len(preferred) - preferred.index(offer.audience)
    if offer.audience == "senior":
        return 1 if customer.is_senior else -5
    return -1


def _rank(
    offer: OfferTemplate,
    hints: tuple[str, ...],
    customer: NormalizedCustomer | None,
) -> tuple[int, ...]:
    """Deterministic sort key, highest first.

    Ordered so each criterion only decides among offers the previous one tied on:

      1. is it relevant to this moment or recipient at all,
      2. fit with this specific recipient (their own past services, senior
         status): personal fit is what "merchant/customer fit" is scored on,
      3. offer shape: a service at a named price beats a bare percentage
         (challenge-brief.md §3 and §11),
      4. how strongly the title matched the moment,
      5. audience fit with the recipient's state.
    """
    keywords = _keyword_score(offer, hints)
    affinity = _customer_affinity(offer, customer)
    return (
        min(keywords + affinity, 1),
        affinity,
        policy.OFFER_TYPE_RANK.get(offer.type, 0),
        keywords,
        _audience_rank(offer, customer),
    )


def select_offer(
    category: NormalizedCategory,
    merchant: NormalizedMerchant,
    trigger: NormalizedTrigger,
    customer: NormalizedCustomer | None = None,
) -> OfferChoice | None:
    """Choose the single offer to reference, preferring one the merchant runs."""
    family = policy.family_for(trigger.kind)
    if family in policy.FAMILIES_WITHOUT_OFFER:
        # An offer bolted onto a compliance notice, a CDE invite or a question to
        # the merchant dilutes the single ask. Decided here rather than at render
        # time so the plan, the rationale and the body always agree.
        return None
    hints = policy.FAMILY_OFFER_HINTS.get(family, ())

    active = list(merchant.active_offers)
    if active:
        best = max(active, key=lambda o: (*_rank(o, hints, customer), o.title))
        matched = _keyword_score(best, hints) + _customer_affinity(best, customer) > 0
        if not matched and family in policy.FAMILIES_NEEDING_RELEVANT_OFFER:
            # Naming an unrelated offer here would read as a bolt-on rather than
            # as part of the reason for writing.
            return None
        return OfferChoice(
            offer=best,
            is_existing=True,
            reason=(
                "merchant's own active offer, matched to this moment"
                if matched
                else "merchant's own active offer"
            ),
        )

    catalog = [o for o in category.offer_catalog if o.title]
    if not catalog:
        return None

    # Only propose a catalog pattern when it is actually relevant to the moment
    # or the recipient: a random catalog entry is noise, not specificity.
    scored = [(o, _rank(o, hints, customer)) for o in catalog]
    relevant = [(o, key) for o, key in scored if key[0] > 0 or key[1] > 0]
    if not relevant:
        return None
    best, _ = max(relevant, key=lambda pair: (*pair[1], pair[0].title))
    return OfferChoice(
        offer=best,
        is_existing=False,
        reason="category catalog pattern — merchant has no live offer to use",
    )
