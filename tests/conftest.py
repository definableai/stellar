"""Fake API keys so adapters construct offline; nothing here ever sends."""

import pytest


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
