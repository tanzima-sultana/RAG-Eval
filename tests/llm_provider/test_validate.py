"""
test_validate.py

Tests for llm_provider.validate.validate_llm. No real API or Ollama server is called.
"""

import pytest

from llm_provider.validate import validate_llm

OLLAMA_MODEL = "ollama_chat/llama3.1:8b"


# ---------- settings every provider shares ----------


@pytest.mark.parametrize("overrides, message", [
    ({"llm_model": "claude-sonnet-5-5"}, "provider"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"timeout_seconds": 0}, "timeout_seconds"),
])
def test_validate_raises_for_invalid_shared_setting(paid_api_allowed, make_llm_args, make_ollama_args, overrides, message):
    """
    Input: LLMArgs with one invalid shared setting.
    Output: ValueError.
    What it does: checks validate_llm names the invalid setting.
    """
    with pytest.raises(ValueError, match=message):
        validate_llm(make_llm_args(**overrides), make_ollama_args())


# ---------- paid models ----------


def test_validate_accepts_paid_model_with_opt_in_key_and_price(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: a paid model in LiteLLM's price map, with ALLOW_PAID_API=1 and its key set.
    Output: nothing raised.
    What it does: checks a usable paid model passes.
    """
    validate_llm(make_llm_args(), make_ollama_args())


def test_validate_raises_for_paid_model_without_allow_paid_api(monkeypatch, make_llm_args, make_ollama_args):
    """
    Input: a paid model with ALLOW_PAID_API not set.
    Output: ValueError.
    What it does: checks a paid model never passes without the opt-in.
    """
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        validate_llm(make_llm_args(), make_ollama_args())


def test_validate_raises_for_paid_model_without_key(monkeypatch, make_llm_args, make_ollama_args):
    """
    Input: a paid model with ALLOW_PAID_API=1 but no ANTHROPIC_API_KEY.
    Output: ValueError.
    What it does: checks the missing key is named.
    """
    monkeypatch.setenv("ALLOW_PAID_API", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        validate_llm(make_llm_args(), make_ollama_args())


def test_validate_raises_for_paid_model_not_in_price_map(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: a paid model LiteLLM has no price for.
    Output: ValueError.
    What it does: checks a model whose cost cannot be computed is rejected.
    """
    with pytest.raises(ValueError, match="price map"):
        validate_llm(make_llm_args(llm_model="anthropic/not-a-real-model"), make_ollama_args())


# ---------- Ollama models ----------


def test_validate_accepts_pulled_ollama_model_without_paid_opt_in(monkeypatch, fake_ollama_server, make_llm_args, make_ollama_args):
    """
    Input: an Ollama model that is pulled, with no ALLOW_PAID_API and no OLLAMA_API_BASE in the environment.
    Output: nothing raised, and the URL the check called.
    What it does: checks a local model needs no opt-in or env variable, and the server URL comes from ollama_args.
    """
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    validate_llm(make_llm_args(llm_model=OLLAMA_MODEL), make_ollama_args())

    assert fake_ollama_server["requested_urls"] == ["http://ollama-test:11434/api/tags"]


def test_validate_raises_when_ollama_server_is_down(fake_ollama_server, make_llm_args, make_ollama_args):
    """
    Input: an Ollama model with the server down.
    Output: ValueError.
    What it does: checks the error names the server URL.
    """
    fake_ollama_server["is_up"] = False

    with pytest.raises(ValueError, match="http://ollama-test:11434"):
        validate_llm(make_llm_args(llm_model=OLLAMA_MODEL), make_ollama_args())


def test_validate_raises_when_ollama_model_is_not_pulled(fake_ollama_server, make_llm_args, make_ollama_args):
    """
    Input: an Ollama model the server has not pulled.
    Output: ValueError.
    What it does: checks the error says how to pull it.
    """
    fake_ollama_server["pulled_models"] = ["mistral:7b"]

    with pytest.raises(ValueError, match="ollama pull llama3.1:8b"):
        validate_llm(make_llm_args(llm_model=OLLAMA_MODEL), make_ollama_args())


def test_validate_raises_for_invalid_ollama_context_length(fake_ollama_server, make_llm_args, make_ollama_args):
    """
    Input: an Ollama model with ollama_context_length 0.
    Output: ValueError.
    What it does: checks the Ollama setting is validated for an Ollama model.
    """
    with pytest.raises(ValueError, match="ollama_context_length"):
        validate_llm(make_llm_args(llm_model=OLLAMA_MODEL), make_ollama_args(ollama_context_length=0))


def test_validate_ignores_ollama_settings_for_paid_model(paid_api_allowed, fake_ollama_server, make_llm_args, make_ollama_args):
    """
    Input: a paid model with an invalid ollama_context_length.
    Output: nothing raised, and no call to the Ollama server.
    What it does: checks Ollama settings are used only for Ollama models.
    """
    validate_llm(make_llm_args(), make_ollama_args(ollama_context_length=0))

    assert fake_ollama_server["requested_urls"] == []
