"""Application configuration.

Everything that identifies this deployment (team name, version, model, contact)
is read from the environment so no challenge-specific literals or secrets are
baked into the code. Values are read once and cached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# Load .env once at import time. Real environment variables always win.
load_dotenv(override=False)


def _env(name: str, default: str) -> str:
    """Return a stripped env var, falling back to ``default`` when unset/blank."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _float(name: str, default: float) -> float:
    """Parse a float env var, falling back to ``default`` when unset or invalid."""
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` when unset or invalid."""
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    """Parse a comma-separated env var into a list of non-empty items."""
    raw = os.getenv(name)
    if not raw:
        return list(default)
    items = [part.strip() for part in raw.split(",")]
    return [item for item in items if item] or list(default)


@dataclass(frozen=True)
class Settings:
    """Immutable view of the process configuration.

    Only non-secret, publicly reportable values live here — `/v1/metadata`
    serialises most of them, so nothing sensitive may be added.
    """

    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Bot identity
    bot_name: str = "Vera"
    bot_version: str = "1.0.0"

    # /v1/metadata fields (see challenge-testing-brief.md §2.5)
    team_name: str = "Vera Rebuild"
    team_members: list[str] = field(default_factory=list)
    model: str = ""
    approach: str = (
        "deterministic 4-context decision engine (trigger ranking, single-signal "
        "selection, real-offer lookup, one CTA) with LLM wording behind a "
        "grounding validator and a deterministic fallback"
    )
    contact_email: str = "unset@example.com"
    submitted_at: str = ""

    # --- LLM wording layer ---------------------------------------------- #
    # The decision engine never needs these: with no provider configured the
    # deterministic renderer writes every message.
    llm_provider: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 4.0
    # Generous enough for a reasoning model to think and still answer; the body
    # itself is a few hundred characters.
    llm_max_tokens: int = 1024
    llm_structured_method: str = "json_schema"
    # Reasoning models (openai/gpt-oss, qwen) spend tokens thinking before they
    # answer. Both are sent only when set, so a non-reasoning model is unaffected:
    #   effort — "low" keeps latency and token spend down; the wording task needs
    #            no deliberation, the decision is already made.
    #   format — "hidden" keeps the reasoning out of the returned content.
    llm_reasoning_effort: str = "low"
    llm_reasoning_format: str = "hidden"
    # Per-tick ceiling: `/v1/tick` has a 10 s budget and may produce up to 20
    # actions, so generation is capped and the rest render deterministically.
    llm_max_calls_per_tick: int = 6
    llm_tick_budget_seconds: float = 8.0

    @property
    def llm_enabled(self) -> bool:
        """True when a provider is configured. The API key is checked separately."""
        return self.llm_provider.strip().lower() in {"groq"}

    @property
    def reported_model(self) -> str:
        """What `/v1/metadata` reports for `model`.

        Derived rather than stored so it cannot drift from the configuration, and
        explicit that the model writes wording rather than making decisions.
        """
        if self.model:
            return self.model
        if self.llm_enabled:
            return f"{self.llm_model} (wording only; decisions are deterministic)"
        return "deterministic rules (no LLM configured)"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the cached Settings instance from the environment."""
    try:
        port = int(_env("PORT", "8000"))
    except ValueError:
        port = 8000

    return Settings(
        app_env=_env("APP_ENV", "development"),
        host=_env("HOST", "0.0.0.0"),
        port=port,
        log_level=_env("LOG_LEVEL", "INFO").upper(),
        bot_name=_env("BOT_NAME", "Vera"),
        bot_version=_env("BOT_VERSION", "1.0.0"),
        team_name=_env("TEAM_NAME", "Vera Rebuild"),
        team_members=_env_list("TEAM_MEMBERS", []),
        model=_env("MODEL_NAME", ""),
        approach=_env(
            "APPROACH",
            "deterministic 4-context decision engine (trigger ranking, single-signal "
            "selection, real-offer lookup, one CTA) with LLM wording behind a "
            "grounding validator and a deterministic fallback",
        ),
        contact_email=_env("CONTACT_EMAIL", "unset@example.com"),
        submitted_at=_env("SUBMITTED_AT", ""),
        llm_provider=_env("LLM_PROVIDER", ""),
        llm_model=_env("LLM_MODEL", "openai/gpt-oss-120b"),
        llm_temperature=_float("LLM_TEMPERATURE", 0.0),
        llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 4.0),
        llm_max_tokens=_int("LLM_MAX_TOKENS", 1024),
        llm_structured_method=_env("LLM_STRUCTURED_METHOD", "json_schema"),
        llm_reasoning_effort=_env("LLM_REASONING_EFFORT", "low"),
        llm_reasoning_format=_env("LLM_REASONING_FORMAT", "hidden"),
        llm_max_calls_per_tick=_int("LLM_MAX_CALLS_PER_TICK", 6),
        llm_tick_budget_seconds=_float("LLM_TICK_BUDGET_SECONDS", 8.0),
    )


def llm_api_key() -> str:
    """Read the provider API key from the environment.

    Deliberately not on `Settings`: the key must never sit on an object that
    `/v1/metadata` serialises, and it is read only at the moment a client is
    built. Never logged, never returned from an endpoint.
    """
    return (os.getenv("GROQ_API_KEY") or "").strip()
