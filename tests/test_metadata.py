"""GET /v1/metadata: bot identity."""

from __future__ import annotations

import json

from config import get_settings

REQUIRED_FIELDS = {
    "team_name",
    "team_members",
    "model",
    "approach",
    "contact_email",
    "version",
    "submitted_at",
}

# Anything matching these must never appear in the response.
SECRET_MARKERS = ("api_key", "apikey", "secret", "token", "password", "bearer", "sk-")


def test_metadata_returns_the_contract_fields(client):
    response = client.get("/v1/metadata")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == REQUIRED_FIELDS
    assert isinstance(body["team_members"], list)
    assert all(isinstance(member, str) for member in body["team_members"])


def test_metadata_version_comes_from_configuration(client):
    assert client.get("/v1/metadata").json()["version"] == get_settings().bot_version


def test_metadata_submitted_at_is_always_populated(client):
    assert client.get("/v1/metadata").json()["submitted_at"]


def test_metadata_leaks_no_secrets(client, monkeypatch):
    # A secret in the environment must not find its way into the response.
    monkeypatch.setenv("VERA_TEST_API_KEY", "sk-should-never-be-exposed")

    serialised = json.dumps(client.get("/v1/metadata").json()).lower()

    assert "sk-should-never-be-exposed" not in serialised
    for marker in SECRET_MARKERS:
        assert marker not in serialised
