"""
test_llm_provider.py

Tests for llm_provider.llm_provider. Calls use LiteLLM's mock_response, so no real API or Ollama server is called.
LiteLLM's mock reply reports 10 input tokens and 20 output tokens.
"""

import litellm
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import LLMResult

from llm_provider.llm_provider import LLMProvider
from llm_provider.llm_provider import TokenCounter

PAID_MODEL = "anthropic/claude-sonnet-5-5"
OLLAMA_MODEL = "ollama_chat/llama3.1:8b"


def set_mock_reply(llm_provider, reply_text):
    """
    Input: llm_provider, an LLMProvider; reply_text, the text the model should reply with.
    Output: nothing.
    What it does: makes every call of the provider's chat model return reply_text through LiteLLM's mock_response.
    """
    llm_provider.chat_model.model_kwargs["mock_response"] = reply_text


# ---------- __init__ ----------


def test_init_validates_settings(monkeypatch, make_llm_args, make_ollama_args):
    """
    Input: a paid model with ALLOW_PAID_API not set.
    Output: ValueError.
    What it does: checks a paid model is never built without the opt-in, even if the layer did not validate first.
    """
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)

    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        LLMProvider(make_llm_args(), make_ollama_args())


def test_init_builds_paid_chat_model_without_ollama_settings(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: a paid model.
    Output: the chat model's settings.
    What it does: checks the model, max_tokens and timeout are passed, and no Ollama URL or context length.
    """
    llm_provider = LLMProvider(make_llm_args(max_tokens=123, timeout_seconds=45), make_ollama_args())

    assert llm_provider.chat_model.model == PAID_MODEL
    assert llm_provider.chat_model.max_tokens == 123
    assert llm_provider.chat_model.request_timeout == 45
    assert llm_provider.chat_model.api_base is None
    assert "num_ctx" not in llm_provider.chat_model.model_kwargs


def test_init_builds_ollama_chat_model_with_ollama_settings(fake_ollama_server, make_llm_args, make_ollama_args):
    """
    Input: an Ollama model.
    Output: the chat model's settings.
    What it does: checks the Ollama URL and context length from ollama_args are passed.
    """
    llm_provider = LLMProvider(make_llm_args(llm_model=OLLAMA_MODEL), make_ollama_args(ollama_context_length=8192))

    assert llm_provider.chat_model.api_base == "http://ollama-test:11434"
    assert llm_provider.chat_model.model_kwargs["num_ctx"] == 8192


# ---------- generate ----------


def test_generate_returns_reply_and_tokens(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: a system prompt and a user prompt, with a mocked reply.
    Output: the LLMResponse.
    What it does: checks the reply text, model and tokens come back.
    """
    llm_provider = LLMProvider(make_llm_args(), make_ollama_args())
    set_mock_reply(llm_provider, "Hello there.")

    llm_response = llm_provider.generate("Be brief.", "Say hello.")

    assert llm_response.text == "Hello there."
    assert llm_response.llm_model == PAID_MODEL
    assert llm_response.input_tokens == 10
    assert llm_response.output_tokens == 20


def test_generate_calls_are_counted(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: two generate calls.
    Output: the token counter.
    What it does: checks every call and its tokens are added up.
    """
    llm_provider = LLMProvider(make_llm_args(), make_ollama_args())
    set_mock_reply(llm_provider, "ok")

    llm_provider.generate("", "first")
    llm_provider.generate("", "second")

    assert llm_provider.token_counter.number_of_calls == 2
    assert llm_provider.token_counter.input_tokens == 20
    assert llm_provider.token_counter.output_tokens == 40


def test_chat_model_calls_are_counted(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: one direct chat_model call, as RAGAS makes them.
    Output: the token counter.
    What it does: checks calls that do not go through generate are counted too.
    """
    llm_provider = LLMProvider(make_llm_args(), make_ollama_args())
    set_mock_reply(llm_provider, "{\"answer\": \"ok\"}")

    llm_provider.chat_model.invoke("Reply in JSON.")

    assert llm_provider.token_counter.number_of_calls == 1
    assert llm_provider.token_counter.input_tokens == 10
    assert llm_provider.token_counter.output_tokens == 20


# ---------- TokenCounter ----------


def test_token_counter_adds_cached_tokens():
    """
    Input: one finished call that reports cache written and cache read tokens.
    Output: the token counter.
    What it does: checks the cached parts are counted next to all input tokens.
    """
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 5,
            "total_tokens": 105,
            "input_token_details": {"cache_creation": 30, "cache_read": 50},
        },
    )
    token_counter = TokenCounter()

    token_counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))

    assert token_counter.number_of_calls == 1
    assert token_counter.input_tokens == 100
    assert token_counter.cache_creation_tokens == 30
    assert token_counter.cache_read_tokens == 50
    assert token_counter.output_tokens == 5


# ---------- compute_cost ----------


def test_compute_cost_uses_litellm_prices(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: one paid call of 10 input and 20 output tokens.
    Output: the LLMCost.
    What it does: checks the cost matches LiteLLM's price map for the model.
    """
    llm_provider = LLMProvider(make_llm_args(), make_ollama_args())
    set_mock_reply(llm_provider, "ok")
    llm_provider.generate("", "hello")

    llm_cost = llm_provider.compute_cost()

    model_prices = litellm.get_model_info(model=PAID_MODEL)
    expected_cost = 10 * model_prices["input_cost_per_token"] + 20 * model_prices["output_cost_per_token"]
    assert llm_cost.total_cost == pytest.approx(expected_cost)
    assert llm_cost.total_cost > 0
    assert llm_cost.number_of_calls == 1
    assert llm_cost.input_tokens == 10
    assert llm_cost.output_tokens == 20


def test_compute_cost_prices_cache_reads_at_cache_price(paid_api_allowed, make_llm_args, make_ollama_args):
    """
    Input: 1000 input tokens, once with none cached and once with all read from the cache.
    Output: the two costs.
    What it does: checks cache reads cost less than normal input tokens.
    """
    llm_provider = LLMProvider(make_llm_args(), make_ollama_args())
    llm_provider.token_counter.input_tokens = 1000
    uncached_cost = llm_provider.compute_cost().total_cost

    llm_provider.token_counter.cache_read_tokens = 1000
    cached_cost = llm_provider.compute_cost().total_cost

    assert cached_cost < uncached_cost


def test_compute_cost_is_zero_for_ollama(fake_ollama_server, make_llm_args, make_ollama_args):
    """
    Input: one Ollama call.
    Output: the LLMCost.
    What it does: checks a local model costs 0.
    """
    llm_provider = LLMProvider(make_llm_args(llm_model=OLLAMA_MODEL), make_ollama_args())
    set_mock_reply(llm_provider, "ok")
    llm_provider.generate("", "hello")

    llm_cost = llm_provider.compute_cost()

    assert llm_cost.total_cost == 0
    assert llm_cost.number_of_calls == 1
