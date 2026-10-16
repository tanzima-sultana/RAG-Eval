"""
test_validate.py

Tests for llm.validate.validate_llm: the checks every provider shares, and
that each provider's own check is called. The provider checks are tested in
the provider's own test file.
"""

import pytest

from llm import validate as validate_module
from llm.validate import validate_llm


@pytest.fixture
def provider_checks(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: a list of the providers whose check was called.
    What it does: replaces the four provider checks, so only the shared checks run.
    """
    called_providers = []

    def make_check(provider):
        """
        Input: provider, the provider name.
        Output: a check that records the provider and returns True.
        What it does: builds one stand-in provider check.
        """
        def check(llm_args):
            """
            Input: llm_args.
            Output: True.
            What it does: records which provider's check ran.
            """
            called_providers.append(provider)
            return True
        return check

    monkeypatch.setattr(validate_module, "validate_claude", make_check("claude"))
    monkeypatch.setattr(validate_module, "validate_chatgpt", make_check("chatgpt"))
    monkeypatch.setattr(validate_module, "validate_gemini", make_check("gemini"))
    monkeypatch.setattr(validate_module, "validate_ollama", make_check("ollama"))
    return called_providers


@pytest.mark.parametrize("provider", ["claude", "chatgpt", "gemini", "ollama"])
def test_valid_settings_run_their_provider_check(provider_checks, make_test_llm_args, provider):
    """
    Input: valid settings for each provider.
    Output: True, and only that provider's check ran.
    What it does: checks validate_llm hands each provider to its own check.
    """
    assert validate_llm(make_test_llm_args(llm=provider)) is True
    assert provider_checks == [provider]


def test_unknown_provider_raises(provider_checks, make_test_llm_args):
    """
    Input: llm "mistral".
    Output: ValueError naming the supported providers, and no provider check ran.
    What it does: checks an unsupported provider is reported.
    """
    with pytest.raises(ValueError, match="not supported"):
        validate_llm(make_test_llm_args(llm="mistral"))
    assert provider_checks == []


@pytest.mark.parametrize("llm_model", ["", "   "])
def test_empty_model_raises(provider_checks, make_test_llm_args, llm_model):
    """
    Input: an empty or whitespace-only llm_model.
    Output: ValueError mentioning llm_model.
    What it does: checks a model name must be given.
    """
    with pytest.raises(ValueError, match="llm_model is empty"):
        validate_llm(make_test_llm_args(llm_model=llm_model))


@pytest.mark.parametrize("input_price, output_price", [(-0.1, 1.0), (1.0, -0.1)])
def test_negative_price_raises(provider_checks, make_test_llm_args, input_price, output_price):
    """
    Input: a negative input or output price.
    Output: ValueError mentioning the prices.
    What it does: checks prices are 0 or more.
    """
    with pytest.raises(ValueError, match="must be 0 or more"):
        validate_llm(make_test_llm_args(input_price=input_price, output_price=output_price))


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_max_tokens_not_positive_raises(provider_checks, make_test_llm_args, max_tokens):
    """
    Input: max_tokens of 0 or below.
    Output: ValueError mentioning max_tokens.
    What it does: checks the output limit is above 0.
    """
    with pytest.raises(ValueError, match="max_tokens"):
        validate_llm(make_test_llm_args(max_tokens=max_tokens))


@pytest.mark.parametrize("timeout_seconds", [0, -5])
def test_timeout_not_positive_raises(provider_checks, make_test_llm_args, timeout_seconds):
    """
    Input: timeout_seconds of 0 or below.
    Output: ValueError mentioning timeout_seconds.
    What it does: checks the call timeout is above 0.
    """
    with pytest.raises(ValueError, match="timeout_seconds"):
        validate_llm(make_test_llm_args(timeout_seconds=timeout_seconds))
