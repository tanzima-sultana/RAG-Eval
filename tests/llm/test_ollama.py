"""
test_ollama.py

Tests for llm.ollama: validate_ollama and OllamaLLM.generate_batch. The Ollama server requests go to
fake_urlopen and the chat model is FakeChatModel, so no server is needed.
"""

import pytest

from llm.llm import TokenCounter
from llm.ollama import OLLAMA_URL
from llm.ollama import OllamaLLM
from llm.ollama import validate_ollama
from llm.schema import LLMResponse

TAGS_BODY = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]}


def make_ollama_llm_args(make_test_llm_args, **overrides):
    """
    Input: the make_test_llm_args fixture; any LLMArgs setting as a keyword argument.
    Output: LLMArgs for llama3.1:8b with every price 0, plus the overrides.
    What it does: builds the settings of the Ollama tests.
    """
    settings = {"llm": "ollama", "llm_model": "llama3.1:8b", "input_price": 0, "output_price": 0, "cache_write_price": 0, "cache_read_price": 0}
    for setting_name in overrides:
        settings[setting_name] = overrides[setting_name]
    return make_test_llm_args(**settings)


# ------ validate_ollama

def test_valid_ollama_settings_pass(no_paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: prices 0, a running server and a pulled model, with no ALLOW_PAID_API.
    Output: True, after one request to the server's model list.
    What it does: checks a local model passes without any paid opt-in.
    """
    fake_urlopen["bodies"] = [TAGS_BODY]

    assert validate_ollama(make_ollama_llm_args(make_test_llm_args)) is True
    assert fake_urlopen["urls"] == [OLLAMA_URL + "/api/tags"]


@pytest.mark.parametrize("price_name", ["input_price", "output_price", "cache_write_price", "cache_read_price"])
def test_ollama_price_above_zero_raises_before_any_request(fake_urlopen, make_test_llm_args, price_name):
    """
    Input: one of the four prices above 0.
    Output: ValueError saying the prices must be 0, and no request sent.
    What it does: checks a local model is never priced, so its cost is always 0.
    """
    with pytest.raises(ValueError, match="prices must be 0"):
        validate_ollama(make_ollama_llm_args(make_test_llm_args, **{price_name: 0.1}))
    assert fake_urlopen["urls"] == []


def test_ollama_server_not_running_raises(fake_urlopen, url_error, make_test_llm_args):
    """
    Input: the server cannot be reached.
    Output: ValueError telling to start it with ollama serve.
    What it does: checks a stopped server is reported with the fix.
    """
    fake_urlopen["error"] = url_error
    with pytest.raises(ValueError, match="ollama serve"):
        validate_ollama(make_ollama_llm_args(make_test_llm_args))


def test_ollama_model_not_pulled_raises(fake_urlopen, make_test_llm_args):
    """
    Input: mistral:7b, which the server does not have.
    Output: ValueError telling to pull it.
    What it does: checks a missing model is reported with the fix.
    """
    fake_urlopen["bodies"] = [TAGS_BODY]
    with pytest.raises(ValueError, match="ollama pull mistral:7b"):
        validate_ollama(make_ollama_llm_args(make_test_llm_args, llm_model="mistral:7b"))


# ------ generate_batch

def test_batch_answers_each_request_one_at_a_time(no_paid_env, fake_chat_model, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: two requests, and a chat model that replies with 5 input and 2 output tokens.
    Output: an LLMResponse per request id, each with its request_id set, and no batch id file.
    What it does: checks Ollama, which has no batch API, sends each request with generate and keys the replies by id.
    """
    ollama_llm = OllamaLLM(make_ollama_llm_args(make_test_llm_args), TokenCounter())
    ollama_llm.chat_model = fake_chat_model("An answer.", {"input_tokens": 5, "output_tokens": 2})
    batch_id_file = tmp_path / "batch_id.txt"

    replies = ollama_llm.generate_batch(batch_requests, str(batch_id_file), 60, 86400)

    assert ollama_llm.chat_model.messages == [
        [("system", "Answer from the context."), ("user", "Context: A\n\nQuestion: a?")],
        [("system", "Answer from the context."), ("user", "Context: B\n\nQuestion: b?")],
    ]
    assert replies["hash_a"] == LLMResponse(llm="ollama", llm_model="llama3.1:8b", request_id="hash_a", text="An answer.",
                                            input_tokens=5, output_tokens=2)
    assert replies["hash_b"].request_id == "hash_b"
    assert not batch_id_file.exists()
