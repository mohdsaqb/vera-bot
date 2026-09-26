"""The wording layer, with the model mocked. No test here makes a network call.

What is pinned: a valid generation is used, and everything else — malformed
output, an exception, a missing key, an ungrounded number, a second ask — resolves
to the deterministic body without raising.
"""

from __future__ import annotations

import pytest

from conftest import FakeWriter
from engine.compose import write_body
from engine.types import GenerationBrief
from services.llm_service import (
    SYSTEM_PROMPT,
    BudgetedWriter,
    DeterministicWriter,
    LLMMessage,
    LLMWriter,
    TickBudget,
    _extract_body,
    build_writer,
    render_prompt,
)


class StubChain:
    """Stands in for the LangChain runnable, returning or raising what it is told."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.last_input: dict | None = None

    def invoke(self, payload, config=None):  # noqa: ARG002 - runnable interface
        self.calls += 1
        self.last_input = payload
        if self.error is not None:
            raise self.error
        return self.result


def _writer_with(monkeypatch, chain, **settings_overrides) -> LLMWriter:
    """An LLMWriter that believes it is configured, wired to a stub chain."""
    from config import Settings

    settings = Settings(
        llm_provider="groq", llm_model="test-model", llm_temperature=0.0,
        **settings_overrides,
    )
    monkeypatch.setattr("services.llm_service.llm_api_key", lambda: "test-key")
    writer = LLMWriter(settings)
    monkeypatch.setattr(writer, "_build_chain", lambda: chain)
    return writer


# --------------------------------------------------------------------------- #
# A. valid structured output
# --------------------------------------------------------------------------- #
def test_valid_structured_output_is_used(monkeypatch, brief):
    grounded = (
        "Dr. Bharat — calls are down 50% this week against a baseline of 12. "
        "Want me to put Aligner Consultation @ ₹499 live?"
    )
    chain = StubChain(LLMMessage(body=grounded, cta=brief.cta))
    writer = _writer_with(monkeypatch, chain)

    result = writer.write(brief)

    assert result.used_llm
    assert result.body == grounded
    assert result.problems == ()
    assert result.model == "test-model"
    assert chain.calls == 1


def test_a_successful_generation_is_logged(monkeypatch, brief, caplog):
    """The only outside signal that the model is really wording messages."""
    import logging

    grounded = (
        "Dr. Bharat — calls are down 50% this week against a baseline of 12. "
        "Want me to put Aligner Consultation @ ₹499 live?"
    )
    writer = _writer_with(monkeypatch, StubChain(LLMMessage(body=grounded, cta=brief.cta)))

    with caplog.at_level(logging.INFO, logger="vera.llm"):
        writer.write(brief)

    lines = [r.getMessage() for r in caplog.records if r.name == "vera.llm"]
    assert any("wording: llm accepted" in line for line in lines), lines
    # The body is not duplicated into the log; only its shape.
    assert not any(grounded in line for line in lines)


def test_one_call_per_message(monkeypatch, brief):
    chain = StubChain(LLMMessage(body=brief.deterministic_body, cta=brief.cta))
    writer = _writer_with(monkeypatch, chain)

    writer.write(brief)
    writer.write(brief)

    assert chain.calls == 2, "one call per message, never a repair round"


def test_a_dict_result_is_also_accepted(monkeypatch, brief):
    chain = StubChain({"body": brief.deterministic_body, "cta": brief.cta})
    writer = _writer_with(monkeypatch, chain)

    assert writer.write(brief).used_llm


# --------------------------------------------------------------------------- #
# B. malformed output
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "result",
    [None, {}, {"cta": "binary_yes_no"}, {"body": None}, "just a string", 42],
)
def test_malformed_output_falls_back(monkeypatch, brief, result):
    writer = _writer_with(monkeypatch, StubChain(result))

    outcome = writer.write(brief)

    assert not outcome.used_llm
    assert outcome.body == brief.deterministic_body
    assert "llm_malformed_response" in outcome.problems


def test_extract_body_handles_every_shape():
    assert _extract_body(LLMMessage(body="x", cta="y")) == "x"
    assert _extract_body({"body": "x"}) == "x"
    assert _extract_body({"nope": 1}) == ""
    assert _extract_body(None) == ""


# --------------------------------------------------------------------------- #
# C / D. exceptions and timeouts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        ConnectionError("no route to host"),
        RuntimeError("provider returned 500"),
        ValueError("bad request"),
    ],
)
def test_provider_failures_fall_back_without_raising(monkeypatch, brief, error):
    writer = _writer_with(monkeypatch, StubChain(error=error))

    outcome = writer.write(brief)

    assert not outcome.used_llm
    assert outcome.body == brief.deterministic_body
    assert any(p.startswith("llm_error:") for p in outcome.problems)


def test_client_that_cannot_be_built_falls_back(monkeypatch, brief):
    from config import Settings

    monkeypatch.setattr("services.llm_service.llm_api_key", lambda: "test-key")
    writer = LLMWriter(Settings(llm_provider="groq"))
    monkeypatch.setattr(writer, "_build_chain", lambda: None)

    assert writer.write(brief).body == brief.deterministic_body


# --------------------------------------------------------------------------- #
# Configuration gates
# --------------------------------------------------------------------------- #
def test_no_provider_configured_means_no_call(brief):
    writer = LLMWriter()

    outcome = writer.write(brief)

    assert not writer.available
    assert outcome.problems == ("llm_provider_not_configured",)
    assert outcome.body == brief.deterministic_body


def test_missing_api_key_means_no_call(monkeypatch, brief):
    from config import Settings

    monkeypatch.setattr("services.llm_service.llm_api_key", lambda: "")
    writer = LLMWriter(Settings(llm_provider="groq"))

    assert writer.unavailable_reason() == "llm_api_key_missing"
    assert writer.write(brief).problems == ("llm_api_key_missing",)


def test_build_writer_defaults_to_deterministic():
    """With no provider configured, no network client is ever constructed."""
    from config import Settings

    assert isinstance(build_writer(Settings()), DeterministicWriter)
    # And through the ambient path, which the autouse fixture has neutralised.
    assert isinstance(build_writer(), DeterministicWriter)


def test_build_writer_returns_an_llm_writer_when_configured():
    from config import Settings

    assert isinstance(build_writer(Settings(llm_provider="groq")), LLMWriter)


def test_a_brief_with_no_fallback_body_produces_nothing(monkeypatch):
    empty = GenerationBrief(
        purpose="outbound", category="Dentists", audience="merchant",
        recipient_name="Meera", salutation="Dr. Meera", tone="peer_clinical",
        language="en", reason="x", facts=(), deterministic_body="",
    )
    writer = _writer_with(monkeypatch, StubChain(LLMMessage(body="anything", cta="none")))

    outcome = writer.write(empty)

    assert outcome.body == ""
    assert outcome.problems == ("no_deterministic_body",)


# --------------------------------------------------------------------------- #
# E-J. ungrounded generations are rejected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "body", "expected"),
    [
        (
            "unsupported number",
            "Dr. Bharat — calls are down 73% this week. Want me to put Aligner "
            "Consultation @ ₹499 live?",
            "unsupported_number:73",
        ),
        (
            "unsupported price",
            "Dr. Bharat — calls are down 50%. Want me to run a ₹149 cleaning special?",
            "unsupported_amount:149",
        ),
        (
            "unsupported date",
            "Dr. Bharat — calls are down 50%. Want me to put Aligner Consultation @ ₹499 "
            "live by 14 Aug?",
            "unsupported_date:14 aug",
        ),
        (
            "hallucinated offer",
            "Dr. Bharat — calls are down 50%. Want me to launch the Monsoon Whitening Bundle?",
            "unsupported_name:Monsoon",
        ),
        (
            "invented competitor",
            "Dr. Bharat — calls are down 50% and Smile Studio is taking them. Want me "
            "to put Aligner Consultation @ ₹499 live?",
            "unsupported_name:Smile",
        ),
        (
            "multiple asks",
            "Dr. Bharat — calls are down 50%. Should I activate the offer and update your listing?",
            "multiple_actions_in_one_ask",
        ),
        (
            "two questions",
            "Dr. Bharat — calls are down 50%. Want me to put Aligner Consultation @ ₹499 "
            "live? Shall I also post it?",
            "multiple_questions",
        ),
        (
            "internal detail",
            "Dr. Bharat — the perf_dip trigger fired. Want me to put Aligner "
            "Consultation @ ₹499 live?",
            "internal_identifier:perf_dip",
        ),
        (
            "admits to being a model",
            "Dr. Bharat — as an AI I can see calls are down 50%. Want me to put Aligner "
            "Consultation @ ₹499 live?",
            "internal_detail:as an ai",
        ),
        (
            "contains a url",
            "Dr. Bharat — calls are down 50%. See https://example.com and reply YES to "
            "put Aligner Consultation @ ₹499 live.",
            "contains_url",
        ),
    ],
)
def test_ungrounded_generations_are_rejected(monkeypatch, brief, label, body, expected):
    writer = _writer_with(monkeypatch, StubChain(LLMMessage(body=body, cta=brief.cta)))

    outcome = writer.write(brief)

    assert not outcome.used_llm, f"{label} should have been rejected"
    assert expected in outcome.problems, f"{label}: got {outcome.problems}"
    assert outcome.body == brief.deterministic_body


def test_empty_generation_is_rejected(monkeypatch, brief):
    writer = _writer_with(monkeypatch, StubChain(LLMMessage(body="   ", cta=brief.cta)))

    assert writer.write(brief).body == brief.deterministic_body


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
class TestModelParameters:
    """Reasoning controls reach the client only when configured."""

    def _built_kwargs(self, monkeypatch, **settings_overrides) -> dict:
        """Capture what ChatGroq would be constructed with."""
        from config import Settings

        captured: dict = {}

        class FakeChatGroq:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def with_structured_output(self, schema, method=None):  # noqa: ARG002
                captured["method"] = method
                return self

        class FakePrompt:
            @staticmethod
            def from_messages(_messages):
                return FakePrompt()

            def __or__(self, other):
                return self

        monkeypatch.setattr("services.llm_service.llm_api_key", lambda: "test-key")
        monkeypatch.setitem(
            __import__("sys").modules, "langchain_groq",
            type("m", (), {"ChatGroq": FakeChatGroq}),
        )
        monkeypatch.setitem(
            __import__("sys").modules, "langchain_core.prompts",
            type("m", (), {"ChatPromptTemplate": FakePrompt}),
        )
        writer = LLMWriter(Settings(llm_provider="groq", **settings_overrides))
        writer._build_chain()
        return captured

    def test_reasoning_controls_are_omitted_when_unset(self, monkeypatch):
        """A model that does not understand them must never receive them."""
        kwargs = self._built_kwargs(
            monkeypatch, llm_reasoning_effort="", llm_reasoning_format=""
        )

        assert "reasoning_effort" not in kwargs
        assert "reasoning_format" not in kwargs

    def test_the_shipped_defaults_do_send_them(self, monkeypatch):
        """The default model is a reasoning model, so the defaults are populated."""
        kwargs = self._built_kwargs(monkeypatch)

        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["reasoning_format"] == "hidden"

    def test_reasoning_controls_are_passed_when_set(self, monkeypatch):
        kwargs = self._built_kwargs(
            monkeypatch, llm_reasoning_effort="low", llm_reasoning_format="hidden"
        )

        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["reasoning_format"] == "hidden"

    def test_the_model_and_determinism_settings_are_passed(self, monkeypatch):
        kwargs = self._built_kwargs(
            monkeypatch, llm_model="openai/gpt-oss-120b", llm_structured_method="json_schema"
        )

        assert kwargs["model"] == "openai/gpt-oss-120b"
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_retries"] == 0, "a hidden retry would blow the tick budget"
        assert kwargs["timeout"] > 0
        assert kwargs["method"] == "json_schema"


class TestPrompt:
    def test_system_prompt_states_the_absolute_rules(self):
        lowered = SYSTEM_PROMPT.lower()
        for rule in (
            "never invent a price", "never invent a date", "never invent a percentage",
            "never invent an offer", "never invent competitor", "never add a second request",
            "do not change who the message is from", "never say or imply that you are an ai",
        ):
            assert rule in lowered, rule

    def test_prompt_carries_the_decision_not_the_dataset(self, brief):
        prompt = render_prompt(brief)

        for fact in brief.facts:
            assert fact in prompt
        assert brief.ask in prompt
        assert brief.cta in prompt
        # Internal decision machinery must not reach the model.
        for leak in ("suppression", "trigger_id", "priority", "merchant.performance",
                     "conversation_id", "evidence"):
            assert leak not in prompt.lower()

    def test_a_suggested_offer_is_labelled_as_not_running(self, brief):
        assert brief.offer_is_live is False
        assert "NOT running" in render_prompt(brief)

    def test_taboo_words_are_passed_through(self, brief):
        prompt = render_prompt(brief)
        assert "NEVER USE THESE WORDS" in prompt
        assert "guaranteed" in prompt


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
class TestBudget:
    def test_budget_stops_calls_after_its_ceiling(self, monkeypatch, brief):
        chain = StubChain(LLMMessage(body=brief.deterministic_body, cta=brief.cta))
        writer = _writer_with(monkeypatch, chain)
        budget = TickBudget(max_calls=2, deadline=float("inf"))

        outcomes = [writer.write(brief, budget=budget) for _ in range(4)]

        assert chain.calls == 2
        assert [o.problems for o in outcomes[2:]] == [
            ("llm_budget_exhausted",), ("llm_budget_exhausted",)
        ]

    def test_an_expired_deadline_stops_calls(self, monkeypatch, brief):
        import time

        chain = StubChain(LLMMessage(body=brief.deterministic_body, cta=brief.cta))
        writer = _writer_with(monkeypatch, chain)
        budget = TickBudget(max_calls=10, deadline=time.monotonic() - 1.0)

        assert writer.write(brief, budget=budget).problems == ("llm_budget_exhausted",)
        assert chain.calls == 0

    def test_a_call_is_refused_when_it_could_outrun_the_deadline(self, monkeypatch, brief):
        """Two 6-second calls must not be started inside an 8-second budget."""
        import time

        chain = StubChain(LLMMessage(body=brief.deterministic_body, cta=brief.cta))
        writer = _writer_with(monkeypatch, chain)
        budget = TickBudget(
            max_calls=10, deadline=time.monotonic() + 5.0, per_call_seconds=6.0
        )

        assert writer.write(brief, budget=budget).problems == ("llm_budget_exhausted",)
        assert chain.calls == 0, "a call that cannot finish in time is not started"

    def test_the_configured_budget_reserves_room_for_one_call(self):
        from config import Settings

        budget = TickBudget.start(
            Settings(llm_tick_budget_seconds=8.0, llm_timeout_seconds=6.0)
        )

        assert budget.per_call_seconds == 6.0
        assert budget.allows(), "a fresh budget must permit the first call"

    def test_budgeted_writer_passes_the_budget_through(self, monkeypatch, brief):
        chain = StubChain(LLMMessage(body=brief.deterministic_body, cta=brief.cta))
        writer = _writer_with(monkeypatch, chain)
        bound = writer.with_budget(TickBudget(max_calls=1, deadline=float("inf")))

        assert isinstance(bound, BudgetedWriter)
        bound.write(brief)
        bound.write(brief)

        assert chain.calls == 1


# --------------------------------------------------------------------------- #
# K / L. the seam compose() uses
# --------------------------------------------------------------------------- #
class TestWriteBody:
    def test_no_writer_means_the_deterministic_body(self, brief):
        assert write_body(brief, None) == (brief.deterministic_body, "deterministic")

    def test_a_valid_generation_is_used(self, brief):
        good = (
            "Dr. Bharat — calls are down 50% week-on-week against a baseline of 12. "
            "Want me to put Aligner Consultation @ ₹499 live?"
        )
        assert write_body(brief, FakeWriter(bodies=[good])) == (good, "llm")

    def test_an_invalid_generation_falls_back(self, brief):
        writer = FakeWriter(bodies=["Calls fell 91%. Want me to run a ₹49 offer?"])

        assert write_body(brief, writer) == (brief.deterministic_body, "deterministic")

    def test_a_writer_that_raises_falls_back(self, brief):
        writer = FakeWriter(raises=RuntimeError("boom"))

        assert write_body(brief, writer) == (brief.deterministic_body, "deterministic")

    def test_deterministic_writer_is_a_passthrough(self, brief):
        assert write_body(brief, DeterministicWriter()) == (
            brief.deterministic_body, "deterministic"
        )
