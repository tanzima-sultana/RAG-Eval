"""
conftest.py

Shared fixtures for the llm tests.

No test sends a request: the list-models and Ollama server calls go to
fake_urlopen, the LangChain chat models are replaced by FakeChatModel, and
the batch clients of the Anthropic, OpenAI and Google SDKs are replaced by
fakes. ALLOW_PAID_API and the API keys are set only inside a test, with
pytest's monkeypatch, so they never reach a real run.
"""

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from common.args import make_llm_args


class FakeHttpResponse:
    """
    Stand-in for the response urllib.request.urlopen returns.
    """

    def __init__(self, body):
        """
        Input: body, a dict sent back as JSON.
        Output: nothing.
        What it does: stores the body.
        """
        self.body = body

    def read(self):
        """
        Input: none.
        Output: the body as JSON bytes.
        What it does: returns the body as the server would.
        """
        return json.dumps(self.body).encode("utf-8")

    def __enter__(self):
        """
        Input: none.
        Output: this response.
        What it does: lets the response be used in a with statement, like the real one.
        """
        return self

    def __exit__(self, exception_type, exception, traceback):
        """
        Input: the exception details, if any.
        Output: False, so an exception is not hidden.
        What it does: closes nothing; the fake holds no connection.
        """
        return False


@pytest.fixture
def fake_urlopen(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: a state dict: "bodies", a list of response bodies returned in order (set by the test); "error", a
    URLError to raise instead (set by the test); and "urls", the URLs asked for.
    What it does: replaces urllib.request.urlopen, so the list-models and Ollama server checks reach no server.
    """
    state = {"bodies": [], "error": None, "urls": [], "headers": []}

    def open_url(url_or_request, timeout):
        """
        Input: a URL string or a urllib Request; timeout.
        Output: a FakeHttpResponse with the next body.
        What it does: records the URL and its headers, then raises the set error or returns the next body.
        """
        if isinstance(url_or_request, urllib.request.Request):
            state["urls"].append(url_or_request.full_url)
            state["headers"].append(dict(url_or_request.header_items()))
        else:
            state["urls"].append(url_or_request)
            state["headers"].append({})
        if state["error"] is not None:
            raise state["error"]
        body = state["bodies"].pop(0)
        return FakeHttpResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    return state


@pytest.fixture
def url_error():
    """
    Input: none.
    Output: a URLError, as urlopen raises when a server cannot be reached.
    What it does: gives tests the error of an unreachable server.
    """
    return urllib.error.URLError("connection refused")


@pytest.fixture
def paid_env(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: nothing.
    What it does: sets ALLOW_PAID_API=1 and fake API keys for this test only. No request is sent with them.
    """
    monkeypatch.setenv("ALLOW_PAID_API", "1")
    monkeypatch.setenv("CLAUDE_API_KEY", "test-claude-key")
    monkeypatch.setenv("CHATGPT_API_KEY", "test-chatgpt-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


@pytest.fixture
def no_paid_env(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: nothing.
    What it does: removes ALLOW_PAID_API and the API keys for this test, whatever the shell has set.
    """
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def make_test_llm_args():
    """
    Input: none.
    Output: a function that builds an LLMArgs.
    What it does: returns a builder where any setting can be given by keyword.
    """
    def build_llm_args(llm="claude", llm_model="claude-haiku-5-5", input_price=1.0, output_price=2.0, max_tokens=100, timeout_seconds=10):
        """
        Input: any LLMArgs setting as a keyword argument.
        Output: an LLMArgs.
        What it does: builds the settings of one LLM.
        """
        return make_llm_args(llm, llm_model, input_price, output_price, max_tokens, timeout_seconds)

    return build_llm_args


class FakeChatModel:
    """
    Stand-in for a LangChain chat model: invoke() records the messages and returns a fixed reply.
    """

    def __init__(self, text, usage_metadata):
        """
        Input: text, the reply text; usage_metadata, the token dict the reply reports, or None.
        Output: nothing.
        What it does: stores the reply.
        """
        self.text = text
        self.usage_metadata = usage_metadata
        self.messages = []

    def invoke(self, messages):
        """
        Input: messages, a list of (role, text) tuples.
        Output: an object with text and usage_metadata, like a LangChain AIMessage.
        What it does: records the messages and returns the reply.
        """
        self.messages.append(messages)
        return SimpleNamespace(text=self.text, usage_metadata=self.usage_metadata)


@pytest.fixture
def fake_chat_model():
    """
    Input: none.
    Output: the FakeChatModel class.
    What it does: lets a test replace an LLM's chat_model.
    """
    return FakeChatModel


@pytest.fixture
def batch_requests():
    """
    Input: none.
    Output: two batch requests, {request_id: {"system_prompt", "user_prompt"}}.
    What it does: gives the batch tests the same requests l9 sends.
    """
    return {
        "hash_a": {"system_prompt": "Answer from the context.", "user_prompt": "Context: A\n\nQuestion: a?"},
        "hash_b": {"system_prompt": "Answer from the context.", "user_prompt": "Context: B\n\nQuestion: b?"},
    }
