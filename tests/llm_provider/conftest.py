"""
conftest.py

Shared fixtures for the llm_provider test suite. No real API or Ollama server is called: paid calls use LiteLLM's
mock_response, and the Ollama server is faked.
"""

import io
import json
import urllib.error

import pytest

from common.args_temp import OllamaArgs
from llm_provider import validate as validate_module
from schema.llm import LLMArgs

PAID_MODEL = "anthropic/claude-sonnet-5-5"
OLLAMA_MODEL = "ollama_chat/llama3.1:8b"


@pytest.fixture
def make_llm_args():
    """
    Input: none.
    Output: a function that builds an LLMArgs.
    What it does: returns a builder with the paid model by default, where any field can be overridden.
    """
    def build_llm_args(**overrides):
        """
        Input: any LLMArgs field as a keyword argument.
        Output: an LLMArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
        field_values["llm_model"] = PAID_MODEL
        field_values["max_tokens"] = 100
        field_values["timeout_seconds"] = 30
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return LLMArgs(**field_values)

    return build_llm_args


@pytest.fixture
def make_ollama_args():
    """
    Input: none.
    Output: a function that builds an OllamaArgs.
    What it does: returns a builder with a test URL and context length, where any field can be overridden.
    """
    def build_ollama_args(**overrides):
        """
        Input: any OllamaArgs field as a keyword argument.
        Output: an OllamaArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
        field_values["ollama_api_base"] = "http://ollama-test:11434"
        field_values["ollama_context_length"] = 32768
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return OllamaArgs(**field_values)

    return build_ollama_args


@pytest.fixture
def paid_api_allowed(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: nothing.
    What it does: sets ALLOW_PAID_API=1 and a fake ANTHROPIC_API_KEY for the test. No real call is made with it.
    """
    monkeypatch.setenv("ALLOW_PAID_API", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")


@pytest.fixture
def fake_ollama_server(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: a dict the test can change: "pulled_models" (model names) and "is_up" (bool). "requested_urls" records calls.
    What it does: replaces urlopen in llm_provider.validate with a fake Ollama /api/tags endpoint.
    """
    server_state = {"pulled_models": ["llama3.1:8b"], "is_up": True, "requested_urls": []}

    def fake_urlopen(url, timeout=None):
        """
        Input: url and timeout, as urlopen takes them.
        Output: a file-like response with the pulled models as JSON.
        What it does: records url, and raises URLError if the server is set as down.
        """
        server_state["requested_urls"].append(url)
        if not server_state["is_up"]:
            raise urllib.error.URLError("connection refused")
        models = []
        for model_name in server_state["pulled_models"]:
            models.append({"name": model_name})
        return io.BytesIO(json.dumps({"models": models}).encode("utf-8"))

    monkeypatch.setattr(validate_module.urllib.request, "urlopen", fake_urlopen)
    return server_state
