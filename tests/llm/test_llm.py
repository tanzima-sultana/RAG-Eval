"""
test_llm.py

Tests for llm.llm (TokenCounter and get_llm) and for what every provider
class shares: generate() returns an LLMResponse and compute_cost() an
LLMCost. The chat models are replaced by FakeChatModel, so no request is sent.
"""

from types import SimpleNamespace

import pytest

from llm.chatgpt import ChatGPTLLM
from llm.claude import ClaudeLLM
from llm.gemini import GeminiLLM
from llm.llm import TokenCounter
from llm.llm import get_llm
from llm.ollama import OllamaLLM
from llm.schema import LLMCost
from llm.schema import LLMResponse

PROVIDERS = [
    ("claude", "claude-haiku-5-5", ClaudeLLM),
    ("chatgpt", "gpt-4.1-nano", ChatGPTLLM),
    ("gemini", "gemini-3.1-flash-lite", GeminiLLM),
    ("ollama", "llama3.1:8b", OllamaLLM),
]


def make_llm_result(usages):
    """
    Input: usages, one usage dict (or None) per reply; None as a whole item means a reply without a message.
    Output: an object shaped like the LangChain result TokenCounter.on_llm_end gets.
    What it does: builds one generation per usage.
    """
    generations = []
    for usage in usages:
        if usage == "no message":
            generations.append(SimpleNamespace())
        else:
            generations.append(SimpleNamespace(message=SimpleNamespace(usage_metadata=usage)))
    return SimpleNamespace(generations=[generations])


# ------ TokenCounter

def test_token_counter_starts_at_zero():
    """
    Input: a new TokenCounter.
    Output: 0 calls, 0 input and 0 output tokens.
    What it does: checks the totals start empty.
    """
    token_counter = TokenCounter()
    assert (token_counter.number_of_calls, token_counter.input_tokens, token_counter.output_tokens) == (0, 0, 0)


def test_token_counter_adds_every_call():
    """
    Input: two calls, the first with two replies.
    Output: 2 calls and the sum of all their tokens.
    What it does: checks the tokens of every reply of every call are added up.
    """
    token_counter = TokenCounter()
    token_counter.on_llm_end(make_llm_result([{"input_tokens": 10, "output_tokens": 2}, {"input_tokens": 5, "output_tokens": 1}]))
    token_counter.on_llm_end(make_llm_result([{"input_tokens": 3, "output_tokens": 4}]))

    assert token_counter.number_of_calls == 2
    assert token_counter.input_tokens == 18
    assert token_counter.output_tokens == 7


def test_token_counter_adds_the_cached_tokens_langchain_reports():
    """
    Input: three replies whose input_token_details hold cache_read 100; cache_creation 50; and Claude's
    ephemeral_5m_input_tokens 30 and ephemeral_1h_input_tokens 20 with cache_creation 0.
    Output: 150 tokens written to the cache, 100 read, and every input token counted once in input_tokens.
    What it does: checks the cache tokens are counted under every key LangChain uses for them.
    """
    token_counter = TokenCounter()
    token_counter.on_llm_end(make_llm_result([
        {"input_tokens": 300, "output_tokens": 1, "input_token_details": {"cache_read": 100}},
        {"input_tokens": 200, "output_tokens": 1, "input_token_details": {"cache_creation": 50, "cache_read": None}},
        {"input_tokens": 100, "output_tokens": 1, "input_token_details": {"cache_creation": 0, "ephemeral_5m_input_tokens": 30,
                                                                          "ephemeral_1h_input_tokens": 20}},
    ]))

    assert token_counter.input_tokens == 600
    assert token_counter.cache_creation_tokens == 100
    assert token_counter.cache_read_tokens == 100


def test_token_counter_skips_replies_without_usage():
    """
    Input: one call with a reply without a message, one with no usage_metadata and one with usage.
    Output: 1 call, counting only the reply with usage.
    What it does: checks a provider that reports no usage does not break the count.
    """
    token_counter = TokenCounter()
    token_counter.on_llm_end(make_llm_result(["no message", None, {"input_tokens": 4, "output_tokens": 2}]))

    assert token_counter.number_of_calls == 1
    assert token_counter.input_tokens == 4
    assert token_counter.output_tokens == 2


# ------ get_llm

@pytest.mark.parametrize("provider, llm_model, llm_class", PROVIDERS)
def test_get_llm_returns_the_provider_class(paid_env, make_test_llm_args, provider, llm_model, llm_class):
    """
    Input: settings for each provider.
    Output: that provider's class, holding the settings and a new TokenCounter.
    What it does: checks get_llm builds the right class. Building one sends no request.
    """
    llm_args = make_test_llm_args(llm=provider, llm_model=llm_model)

    llm_object = get_llm(llm_args)

    assert isinstance(llm_object, llm_class)
    assert llm_object.llm_args == llm_args
    assert llm_object.token_counter.number_of_calls == 0


def test_get_llm_gives_each_llm_its_own_token_counter(paid_env, make_test_llm_args):
    """
    Input: two get_llm calls with the same settings.
    Output: two different TokenCounter objects.
    What it does: checks the tokens of one layer's LLM are never added to another's.
    """
    first_llm = get_llm(make_test_llm_args())
    second_llm = get_llm(make_test_llm_args())
    assert first_llm.token_counter is not second_llm.token_counter


def test_get_llm_unknown_provider_raises(make_test_llm_args):
    """
    Input: llm "mistral".
    Output: ValueError saying it is not supported.
    What it does: checks an unsupported provider is reported.
    """
    with pytest.raises(ValueError, match="not supported"):
        get_llm(make_test_llm_args(llm="mistral"))


@pytest.mark.parametrize("provider, llm_model, llm_class", PROVIDERS[:3])
def test_paid_llm_is_never_built_without_allow_paid_api(no_paid_env, make_test_llm_args, provider, llm_model, llm_class):
    """
    Input: settings for each paid provider, with ALLOW_PAID_API not set.
    Output: ValueError mentioning ALLOW_PAID_API.
    What it does: checks a paid LLM cannot be built even by code that skipped validate_llm.
    """
    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        get_llm(make_test_llm_args(llm=provider, llm_model=llm_model))


def test_ollama_is_built_without_allow_paid_api(no_paid_env, make_test_llm_args):
    """
    Input: Ollama settings, with ALLOW_PAID_API not set.
    Output: an OllamaLLM.
    What it does: checks the local, free provider needs no opt-in.
    """
    assert isinstance(get_llm(make_test_llm_args(llm="ollama", llm_model="llama3.1:8b", input_price=0, output_price=0)), OllamaLLM)


# ------ generate and compute_cost, the same for every provider

@pytest.mark.parametrize("provider, llm_model, llm_class", PROVIDERS)
def test_generate_sends_system_and_user_prompt_and_returns_a_response(paid_env, make_test_llm_args, fake_chat_model, provider, llm_model, llm_class):
    """
    Input: a system prompt and a user prompt, and a chat model that replies with 12 input and 3 output tokens.
    Output: an LLMResponse with the reply text, the tokens, the provider and model, and request_id "".
    What it does: checks generate sends both prompts in order and wraps the reply.
    """
    llm_object = get_llm(make_test_llm_args(llm=provider, llm_model=llm_model))
    llm_object.chat_model = fake_chat_model("Chris Allen is the District Manager.", {"input_tokens": 12, "output_tokens": 3})

    llm_response = llm_object.generate("Answer from the context.", "Who is Chris Allen?")

    assert llm_object.chat_model.messages == [[("system", "Answer from the context."), ("user", "Who is Chris Allen?")]]
    assert llm_response == LLMResponse(llm=provider, llm_model=llm_model, request_id="", text="Chris Allen is the District Manager.",
                                       input_tokens=12, output_tokens=3)


@pytest.mark.parametrize("provider, llm_model, llm_class", PROVIDERS)
def test_generate_with_empty_system_prompt_sends_only_the_user_prompt(paid_env, make_test_llm_args, fake_chat_model, provider, llm_model, llm_class):
    """
    Input: system prompt "", and a chat model that reports no usage.
    Output: only the user message is sent, and the response has 0 tokens.
    What it does: checks l5's prompts without instructions, and a provider that reports no tokens.
    """
    llm_object = get_llm(make_test_llm_args(llm=provider, llm_model=llm_model))
    llm_object.chat_model = fake_chat_model("A rewritten question?", None)

    llm_response = llm_object.generate("", "Rewrite: who?")

    assert llm_object.chat_model.messages == [[("user", "Rewrite: who?")]]
    assert llm_response.text == "A rewritten question?"
    assert (llm_response.input_tokens, llm_response.output_tokens) == (0, 0)


@pytest.mark.parametrize("provider, llm_model, llm_class", PROVIDERS)
def test_compute_cost_returns_the_cost_of_the_counted_tokens(paid_env, make_test_llm_args, capsys, provider, llm_model, llm_class):
    """
    Input: 3 calls, 2,000,000 input tokens of which 300,000 written to the cache and 500,000 read from it, and 500,000
    output tokens; prices $1.00 input, $2.00 output, $1.25 cache write and $0.10 cache read per million.
    Output: an LLMCost of $1.20 for the 1,200,000 uncached input tokens, $0.375 cache write, $0.05 cache read, $1.00
    output and $2.625 in total, also printed.
    What it does: checks each kind of token is priced at its own price, for every provider.
    """
    llm_object = get_llm(make_test_llm_args(llm=provider, llm_model=llm_model, input_price=1.0, output_price=2.0,
                                            cache_write_price=1.25, cache_read_price=0.1))
    llm_object.token_counter.number_of_calls = 3
    llm_object.token_counter.input_tokens = 2000000
    llm_object.token_counter.cache_creation_tokens = 300000
    llm_object.token_counter.cache_read_tokens = 500000
    llm_object.token_counter.output_tokens = 500000

    llm_cost = llm_object.compute_cost()

    assert llm_cost == LLMCost(llm=provider, llm_model=llm_model, number_of_calls=3, input_tokens=2000000,
                               cache_creation_tokens=300000, cache_read_tokens=500000, output_tokens=500000,
                               input_cost=pytest.approx(1.2), cache_write_cost=pytest.approx(0.375),
                               cache_read_cost=pytest.approx(0.05), output_cost=pytest.approx(1.0), total_cost=pytest.approx(2.625))
    printed = capsys.readouterr().out
    assert f"{llm_model} cache read tokens : 500000" in printed
    assert f"{llm_model} cost : $2.6250" in printed
