"""Deterministic formatting of the numbers that appear in messages.

Kept separate because both the signal extractors (which build fact sentences)
and the renderer (which builds bodies) need the exact same phrasing, and because
"how a number is written" is where vague copy usually creeps in. Nothing here
computes new values — it only presents values read from the contexts.
"""

from __future__ import annotations

from datetime import datetime

from engine.normalize import parse_iso

_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def pct(fraction: float, *, signed: bool = False) -> str:
    """Format a 0-1 fraction as a whole-number percentage: -0.5 -> "50%"."""
    value = fraction * 100
    rounded = round(abs(value)) if abs(value) >= 1 else round(abs(value), 1)
    text = f"{rounded:g}%"
    if signed and value > 0:
        return f"+{text}"
    if signed and value < 0:
        return f"-{text}"
    return text


def rate_pct(fraction: float) -> str:
    """Format a rate such as CTR for display: 0.021 -> "2.1%"."""
    return f"{round(fraction * 100, 1):g}%"


def points(gap: float) -> str:
    """Format a percentage-point gap: 0.009 -> "0.9 percentage points"."""
    value = round(abs(gap) * 100, 1)
    unit = "percentage point" if value == 1 else "percentage points"
    return f"{value:g} {unit}"


def count(value: float) -> str:
    """Format a count with thousands separators, dropping a trailing .0."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def rupees(value: str | float | int) -> str:
    """Format a rupee amount as ₹1,420. Passes through text that isn't numeric."""
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("₹", "").strip()
        if not cleaned.replace(".", "", 1).isdigit():
            return value
        value = float(cleaned)
    return f"₹{count(value)}"


def day_label(value: str | datetime | None) -> str:
    """Format a date as "28 Apr" — short, unambiguous, no invented weekday."""
    moment = parse_iso(value)
    if moment is None:
        return ""
    return f"{moment.day} {_MONTHS[moment.month - 1]}"


def clock(iso_text: str) -> str:
    """Render the local wall-clock time of an ISO string as "7:30pm".

    Reads the characters of the supplied string rather than converting zones, so
    the time shown is the one the payload stated in its own offset.
    """
    text = str(iso_text)
    if len(text) < 16 or text[10] not in "T ":
        return ""
    try:
        hour, minute = int(text[11:13]), int(text[14:16])
    except ValueError:
        return ""
    suffix = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d}{suffix}" if minute else f"{hour12}{suffix}"


def first_sentence(text: str) -> str:
    """First sentence of a summary, without splitting on abbreviations.

    A period only ends a sentence when it follows a word of more than two
    characters and precedes a capitalised word — which keeps "Dr. R. Mehta" and
    "1.5 mSv" intact in the digest summaries.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    for index in range(len(cleaned) - 2):
        if cleaned[index] != ".":
            continue
        word = cleaned[:index].rsplit(" ", 1)[-1]
        if len(word) <= 2 or word[-1].isdigit():
            continue
        rest = cleaned[index + 1 :]
        if rest.startswith(" ") and len(rest) > 1 and rest[1].isupper():
            return cleaned[:index]
    return cleaned.rstrip(".")


def direction(fraction: float) -> str:
    """"up" or "down" for a signed fraction."""
    return "up" if fraction >= 0 else "down"


def join_terms(terms: list[str] | tuple[str, ...], *, last: str = "and") -> str:
    """Join terms for prose: ("a","b","c") -> "a, b and c"."""
    items = [t for t in terms if t]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {last} {items[-1]}"
