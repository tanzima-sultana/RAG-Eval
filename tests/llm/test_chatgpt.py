"""
test_chatgpt.py

Tests for llm.chatgpt: validate_chatgpt and ChatGPTLLM.generate_batch. The list-models request goes to
fake_urlopen and the OpenAI batch client is FakeOpenAI, so nothing is sent.
"""

import json
from types import SimpleNamespace

import pytest

from llm import chatgpt as chatgpt_module
from llm.chatgpt import ChatGPTLLM
from llm.chatgpt import validate_chatgpt
from llm.llm import TokenCounter
from llm.schema import LLMResponse

MODELS_BODY = {"data": [{"id": "gpt-4.1-mini"}, {"id": "gpt-4.1-nano"}]}


# ------ validate_chatgpt

def test_valid_chatgpt_settings_pass(paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: ALLOW_PAID_API=1, a key, a listed model and prices above 0.
    Output: True, after one list-models request sent with the key.
    What it does: checks a usable ChatGPT model passes.
    """
    fake_urlopen["bodies"] = [MODELS_BODY]

    assert validate_chatgpt(make_test_llm_args(llm="chatgpt", llm_model="gpt-4.1-nano")) is True
    assert fake_urlopen["urls"] == ["https://api.openai.com/v1/models"]
    assert fake_urlopen["headers"][0]["Authorization"] == "Bearer test-chatgpt-key"


def test_chatgpt_without_allow_paid_api_raises_before_any_request(no_paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: ALLOW_PAID_API not set.
    Output: ValueError mentioning ALLOW_PAID_API, and no request sent.
    What it does: checks a paid run must opt in before anything else.
    """
    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        validate_chatgpt(make_test_llm_args(llm="chatgpt", llm_model="gpt-4.1-nano"))
    assert fake_urlopen["urls"] == []


def test_chatgpt_without_key_raises(paid_env, monkeypatch, fake_urlopen, make_test_llm_args):
    """
    Input: CHATGPT_API_KEY not set.
    Output: ValueError mentioning CHATGPT_API_KEY.
    What it does: checks the key is required.
    """
    monkeypatch.delenv("CHATGPT_API_KEY")
    with pytest.raises(ValueError, match="CHATGPT_API_KEY"):
        validate_chatgpt(make_test_llm_args(llm="chatgpt", llm_model="gpt-4.1-nano"))


def test_chatgpt_model_not_listed_raises(paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: gpt-0, which OpenAI does not list.
    Output: ValueError saying it is not one of the OpenAI models.
    What it does: checks a misspelled or retired model is caught before any paid call.
    """
    fake_urlopen["bodies"] = [MODELS_BODY]
    with pytest.raises(ValueError, match="not one of the OpenAI models"):
        validate_chatgpt(make_test_llm_args(llm="chatgpt", llm_model="gpt-0"))


def test_chatgpt_models_not_reachable_raises(paid_env, fake_urlopen, url_error, make_test_llm_args):
    """
    Input: the list-models request fails.
    Output: ValueError saying the models could not be listed.
    What it does: checks a network failure is reported, not ignored.
    """
    fake_urlopen["error"] = url_error
    with pytest.raises(ValueError, match="Could not list the ChatGPT models"):
        validate_chatgpt(make_test_llm_args(llm="chatgpt", llm_model="gpt-4.1-nano"))


@pytest.mark.parametrize("input_price, output_price", [(0, 2.0), (1.0, 0)])
def test_chatgpt_zero_price_raises(paid_env, fake_urlopen, make_test_llm_args, input_price, output_price):
    """
    Input: an input or output price of 0.
    Output: ValueError saying the model needs prices above 0.
    What it does: checks every paid run can compute its cost.
    """
    fake_urlopen["bodies"] = [MODELS_BODY]
    with pytest.raises(ValueError, match="price above 0"):
        validate_chatgpt(make_test_llm_args(llm="chatgpt", llm_model="gpt-4.1-nano", input_price=input_price, output_price=output_price))


# ------ generate_batch

class FakeOpenAI:
    """
    Stand-in for openai.OpenAI: the files and batches clients of a batch that ends after a set number of checks.
    """

    def __init__(self, state):
        """
        Input: state, the test's dict of settings and records.
        Output: nothing.
        What it does: builds files (create, content) and batches (create, retrieve).
        """
        self.state = state
        self.files = SimpleNamespace(create=self.create_file, content=self.file_content)
        self.batches = SimpleNamespace(create=self.create_batch, retrieve=self.retrieve_batch)

    def create_file(self, file, purpose):
        """
        Input: file, a (name, bytes) pair; purpose.
        Output: a file with id file_1.
        What it does: records the uploaded JSONL lines and the purpose.
        """
        self.state["uploaded_lines"] = file[1].decode("utf-8").strip().split("\n")
        self.state["purpose"] = purpose
        return SimpleNamespace(id="file_1")

    def create_batch(self, input_file_id, endpoint, completion_window):
        """
        Input: input_file_id, endpoint and completion_window.
        Output: a batch with id batch_1.
        What it does: records the batch settings.
        """
        self.state["created"].append((input_file_id, endpoint, completion_window))
        return SimpleNamespace(id="batch_1")

    def retrieve_batch(self, batch_id):
        """
        Input: batch_id.
        Output: the batch, "in_progress" until checks_before_end checks have been made, then state["end_status"].
        What it does: records the check.
        """
        self.state["retrieved"].append(batch_id)
        if len(self.state["retrieved"]) > self.state["checks_before_end"]:
            return SimpleNamespace(status=self.state["end_status"], output_file_id=self.state["output_file_id"],
                                   request_counts=SimpleNamespace(completed=2))
        return SimpleNamespace(status="in_progress", output_file_id=None, request_counts=SimpleNamespace(completed=0))

    def file_content(self, file_id):
        """
        Input: file_id, the output file.
        Output: an object whose text is one JSONL line per entry in state["results"].
        What it does: builds the output file the way OpenAI writes it.
        """
        lines = []
        for custom_id, status_code, text, input_tokens, output_tokens in self.state["results"]:
            body = {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}}
            lines.append(json.dumps({"custom_id": custom_id, "response": {"status_code": status_code, "body": body}}))
        return SimpleNamespace(text="\n".join(lines) + "\n")


@pytest.fixture
def fake_openai(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the state dict FakeOpenAI uses; the test sets "results", "checks_before_end", "end_status" and
    "output_file_id".
    What it does: makes generate_batch use FakeOpenAI. ChatOpenAI keeps the real SDK, untouched.
    """
    state = {"created": [], "retrieved": [], "checks_before_end": 0, "end_status": "completed", "output_file_id": "out_1",
             "results": [], "keys": []}

    def make_client(api_key, timeout):
        """
        Input: api_key and timeout, as openai.OpenAI takes them.
        Output: a FakeOpenAI.
        What it does: records the key and builds the fake client.
        """
        state["keys"].append(api_key)
        return FakeOpenAI(state)

    monkeypatch.setattr(chatgpt_module, "openai", SimpleNamespace(OpenAI=make_client))
    return state


def make_chatgpt_llm(make_test_llm_args):
    """
    Input: the make_test_llm_args fixture.
    Output: a ChatGPTLLM for gpt-4.1-nano.
    What it does: builds the LLM the batch tests call.
    """
    return ChatGPTLLM(make_test_llm_args(llm="chatgpt", llm_model="gpt-4.1-nano"), TokenCounter())


def test_batch_uploads_every_request_once_and_returns_responses(paid_env, fake_openai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: two requests; the batch completes after one in_progress check.
    Output: an LLMResponse per request id with its text and tokens, the tokens counted, and the batch id saved.
    What it does: checks one JSONL line per request is uploaded for the chat completions endpoint, and the replies
    come back by custom_id.
    """
    fake_openai["checks_before_end"] = 1
    fake_openai["results"] = [("hash_b", 200, "Answer B.", 30, 4), ("hash_a", 200, "Answer A.", 20, 3)]
    batch_id_file = str(tmp_path / "generated" / "batch_id.txt")
    chatgpt_llm = make_chatgpt_llm(make_test_llm_args)

    replies = chatgpt_llm.generate_batch(batch_requests, batch_id_file, 0, 60)

    assert fake_openai["keys"] == ["test-chatgpt-key"]
    assert fake_openai["purpose"] == "batch"
    assert fake_openai["created"] == [("file_1", "/v1/chat/completions", "24h")]
    first_line = json.loads(fake_openai["uploaded_lines"][0])
    assert first_line == {
        "custom_id": "hash_a", "method": "POST", "url": "/v1/chat/completions",
        "body": {"model": "gpt-4.1-nano", "max_completion_tokens": 100,
                 "messages": [{"role": "system", "content": "Answer from the context."},
                              {"role": "user", "content": "Context: A\n\nQuestion: a?"}]},
    }
    assert len(fake_openai["uploaded_lines"]) == 2
    assert replies["hash_a"] == LLMResponse(llm="chatgpt", llm_model="gpt-4.1-nano", request_id="hash_a", text="Answer A.",
                                            input_tokens=20, output_tokens=3)
    assert replies["hash_b"].text == "Answer B."
    assert chatgpt_llm.token_counter.number_of_calls == 2
    assert chatgpt_llm.token_counter.input_tokens == 50
    assert chatgpt_llm.token_counter.output_tokens == 7
    with open(batch_id_file, encoding="utf-8") as f:
        assert f.read() == "batch_1"


def test_batch_resumes_a_saved_batch_without_uploading_again(paid_env, fake_openai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a batch id file from an earlier run.
    Output: the replies of that batch, with no file uploaded and no batch created.
    What it does: checks a stopped run resumes its batch instead of paying for the requests again.
    """
    fake_openai["results"] = [("hash_a", 200, "Answer A.", 20, 3)]
    batch_id_file = tmp_path / "batch_id.txt"
    batch_id_file.write_text("batch_earlier")

    replies = make_chatgpt_llm(make_test_llm_args).generate_batch(batch_requests, str(batch_id_file), 0, 60)

    assert fake_openai["created"] == []
    assert "uploaded_lines" not in fake_openai
    assert fake_openai["retrieved"] == ["batch_earlier"]
    assert list(replies) == ["hash_a"]


def test_batch_timeout_raises_and_keeps_the_batch_id(paid_env, fake_openai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a batch that never ends, and a timeout of 0 seconds.
    Output: TimeoutError saying to run again, with the batch id file kept.
    What it does: checks the wait is limited and the next run can resume the same batch.
    """
    fake_openai["checks_before_end"] = 1000
    batch_id_file = tmp_path / "batch_id.txt"

    with pytest.raises(TimeoutError, match="Run again to resume it"):
        make_chatgpt_llm(make_test_llm_args).generate_batch(batch_requests, str(batch_id_file), 0, 0)

    assert batch_id_file.read_text() == "batch_1"


def test_batch_leaves_out_failed_requests(paid_env, fake_openai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: hash_a answered with status 200, hash_b with status 500.
    Output: a reply for hash_a only, and only its tokens counted.
    What it does: checks a failed request gets no answer, so l9 can send it again.
    """
    fake_openai["results"] = [("hash_a", 200, "Answer A.", 20, 3), ("hash_b", 500, "", 0, 0)]

    chatgpt_llm = make_chatgpt_llm(make_test_llm_args)
    replies = chatgpt_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    assert list(replies) == ["hash_a"]
    assert chatgpt_llm.token_counter.number_of_calls == 1


def test_failed_batch_without_output_file_returns_no_replies(paid_env, fake_openai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a batch that ends "failed" with no output file.
    Output: no replies, and no tokens counted.
    What it does: checks a failed batch does not crash, so l9 reports every request as failed.
    """
    fake_openai["end_status"] = "failed"
    fake_openai["output_file_id"] = None

    chatgpt_llm = make_chatgpt_llm(make_test_llm_args)
    replies = chatgpt_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    assert replies == {}
    assert chatgpt_llm.token_counter.number_of_calls == 0
