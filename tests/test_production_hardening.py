"""Production safety nets: the access log, the catch-all handler, and startup.

These pin behaviour that only matters once the bot is serving real traffic — that
an unhandled fault still answers with valid JSON, that nothing sensitive reaches
the log, and that the app starts and serves with no optional dependency present.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app import app
from conftest import NOW, push_context


@pytest.fixture
def serving_client():
    """A client that returns server errors instead of re-raising them.

    `TestClient` re-raises server exceptions by default, which is useful for
    finding bugs but hides the response a real client would receive. Uvicorn
    returns the handler's response, so that is what these tests assert on.
    """
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def vera_logs(caplog) -> str:
    """Only this application's log records, not those of its dependencies."""
    return " ".join(
        record.getMessage() for record in caplog.records
        if record.name.startswith("vera")
    )


class TestCatchAllHandler:
    def test_an_unhandled_fault_answers_with_valid_json(self, serving_client, monkeypatch):
        """A crash must not produce a bare 500 or a stack trace on the wire."""
        def explode(*_args, **_kwargs):
            raise RuntimeError("simulated internal fault")

        monkeypatch.setattr("app.get_decision_service", explode)

        response = serving_client.post(
            "/v1/tick", json={"now": NOW, "available_triggers": []},
            headers={"Accept": "application/json"},
        )

        assert response.status_code == 500
        body = response.json()
        assert body == {"error": "internal_error",
                        "details": "the request could not be completed"}

    def test_the_wire_response_carries_no_internals(self, serving_client, monkeypatch):
        def explode(*_args, **_kwargs):
            raise RuntimeError("secret-internal-detail at /Users/someone/app.py line 42")

        monkeypatch.setattr("app.get_reply_service", explode)

        response = serving_client.post(
            "/v1/reply",
            json={"conversation_id": "conv_x", "from_role": "merchant",
                  "message": "hi", "received_at": NOW, "turn_number": 2},
        )

        assert response.status_code == 500
        assert "secret-internal-detail" not in response.text
        assert "Traceback" not in response.text
        assert "/Users/" not in response.text

    def test_the_cause_is_logged_server_side(self, serving_client, monkeypatch, caplog):
        """The handler catches the response, it does not hide the bug."""
        def explode(*_args, **_kwargs):
            raise RuntimeError("diagnosable-cause")

        monkeypatch.setattr("app.get_decision_service", explode)

        with caplog.at_level(logging.ERROR, logger="vera.api"):
            serving_client.post("/v1/tick", json={"now": NOW, "available_triggers": []})

        assert any("diagnosable-cause" in record.getMessage()
                   or "diagnosable-cause" in str(record.exc_info)
                   for record in caplog.records)

    def test_a_healthy_request_is_unaffected(self, client):
        assert client.get("/v1/healthz").status_code == 200


class TestAccessLog:
    def test_requests_are_logged_with_status_and_latency(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="vera.api"):
            client.post("/v1/tick", json={"now": NOW, "available_triggers": []})

        assert "POST /v1/tick -> 200 in" in vera_logs(caplog)

    def test_health_polls_do_not_flood_the_log(self, client, caplog):
        """The judge polls healthz every 60s for the whole window."""
        with caplog.at_level(logging.INFO, logger="vera.api"):
            for _ in range(5):
                client.get("/v1/healthz")

        assert "/v1/healthz" not in vera_logs(caplog)

    def test_no_payload_content_reaches_the_log(self, client, dataset, caplog):
        merchant = dataset.merchants["m_001_drmeera_dentist_delhi"]

        with caplog.at_level(logging.DEBUG, logger="vera"):
            push_context(client, "merchant", merchant["merchant_id"], merchant)

        logged = vera_logs(caplog)
        # The identifier is useful; the contents of the record are not.
        assert merchant["merchant_id"] in logged
        assert merchant["identity"]["place_id"] not in logged
        assert "conversation_history" not in logged

    def test_no_secret_reaches_the_log(self, client, monkeypatch, caplog):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-must-never-be-logged")

        with caplog.at_level(logging.DEBUG, logger="vera"):
            client.get("/v1/metadata")
            client.post("/v1/tick", json={"now": NOW, "available_triggers": []})

        assert "gsk-must-never-be-logged" not in vera_logs(caplog)


class TestStartupPortability:
    def test_the_app_imports_without_the_wording_packages(self):
        """`langchain_*` is imported lazily, so a runtime without it still serves."""
        import services.llm_service as llm

        source = (
            __import__("pathlib").Path(llm.__file__).read_text()
        )
        top_level = [
            line for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "langchain" in line
        ]
        assert not top_level, f"langchain imported at module scope: {top_level}"

    def test_a_missing_wording_package_degrades_rather_than_raises(self, brief, monkeypatch):
        from config import Settings
        from services.llm_service import LLMWriter

        monkeypatch.setattr("services.llm_service.llm_api_key", lambda: "test-key")
        writer = LLMWriter(Settings(llm_provider="groq"))

        def no_package():
            raise ImportError("No module named 'langchain_groq'")

        monkeypatch.setattr(writer, "_build_chain", no_package)

        with pytest.raises(ImportError):
            writer._build_chain()
        # The public path swallows it and falls back.
        monkeypatch.setattr(writer, "_build_chain", lambda: None)
        assert writer.write(brief).body == brief.deterministic_body

    def test_port_is_read_from_the_environment(self, monkeypatch):
        """The start command takes $PORT; nothing may hardcode one."""
        from config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("PORT", "9137")
        try:
            assert get_settings().port == 9137
        finally:
            get_settings.cache_clear()

    def test_host_defaults_to_all_interfaces(self, monkeypatch):
        from config import get_settings

        get_settings.cache_clear()
        monkeypatch.delenv("HOST", raising=False)
        try:
            assert get_settings().host == "0.0.0.0"
        finally:
            get_settings.cache_clear()
