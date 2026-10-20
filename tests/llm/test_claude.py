"""
test_claude.py

Tests for llm.claude: validate_claude and ClaudeLLM.generate_batch. The list-models request goes to
fake_urlopen and the Anthropic batch client is FakeAnthropic, so nothing is sent.
"""

from types import SimpleNamespace

import pytest

from llm import claude as claude_module
from llm.claude import ClaudeLLM
from llm.claude import validate_claude
from llm.llm import TokenCounter
from llm.schema import LLMResponse

MODELS_BODY = {"data": [{"id": "claude-sonnet-5-5"}, {"id": "claude-haiku-5-5"}]}


# ------ validate_claude

def test_valid_claude_settings_pass(paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: ALLOW_PAID_API=1, a key, a listed model and prices above 0.
    Output: True, after one list-models request sent with the key.
    What it does: checks a usable Claude model passes.
    """
    fake_urlopen["bodies"] = [MODELS_BODY]

    assert validate_claude(make_test_llm_args()) is True
    assert fake_urlopen["urls"] == ["https://api.anthropic.com/v1/models?limit=1000"]
    assert fake_urlopen["headers"][0]["X-api-key"] == "test-claude-key"


def test_claude_without_allow_paid_api_raises_before_any_request(no_paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: ALLOW_PAID_API not set.
    Output: ValueError mentioning ALLOW_PAID_API, and no request sent.
    What it does: checks a paid run must opt in before anything else.
    """
    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        validate_claude(make_test_llm_args())
    assert fake_urlopen["urls"] == []


def test_claude_without_key_raises(paid_env, monkeypatch, fake_urlopen, make_test_llm_args):
    """
    Input: CLAUDE_API_KEY not set.
    Output: ValueError mentioning CLAUDE_API_KEY.
    What it does: checks the key is required.
    """
    monkeypatch.delenv("CLAUDE_API_KEY")
    with pytest.raises(ValueError, match="CLAUDE_API_KEY"):
        validate_claude(make_test_llm_args())


def test_claude_model_not_listed_raises(paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: claude-opus-0, which Anthropic does not list.
    Output: ValueError saying it is not one of the Anthropic models.
    What it does: checks a misspelled or retired model is caught before any paid call.
    """
    fake_urlopen["bodies"] = [MODELS_BODY]
    with pytest.raises(ValueError, match="not one of the Anthropic models"):
        validate_claude(make_test_llm_args(llm_model="claude-opus-0"))


def test_claude_models_not_reachable_raises(paid_env, fake_urlopen, url_error, make_test_llm_args):
    """
    Input: the list-models request fails.
    Output: ValueError saying the models could not be listed.
    What it does: checks a network failure is reported, not ignored.
    """
    fake_urlopen["error"] = url_error
    with pytest.raises(ValueError, match="Could not list the Claude models"):
        validate_claude(make_test_llm_args())


@pytest.mark.parametrize("price_name", ["input_price", "output_price", "cache_write_price", "cache_read_price"])
def test_claude_zero_price_raises(paid_env, fake_urlopen, make_test_llm_args, price_name):
    """
    Input: one of the four prices at 0.
    Output: ValueError saying the model needs prices above 0.
    What it does: checks every paid run can compute its cost.
    """
    fake_urlopen["bodies"] = [MODELS_BODY]
    with pytest.raises(ValueError, match="price above 0"):
        validate_claude(make_test_llm_args(**{price_name: 0}))


# ------ generate_batch

class FakeAnthropic:
    """
    Stand-in for anthropic.Anthropic: a Message Batches client that ends after a set number of status checks.
    """

    def __init__(self, state):
        """
        Input: state, the test's dict of settings and records.
        Output: nothing.
        What it does: builds messages.batches with create, retrieve and results.
        """
        self.state = state
        self.messages = SimpleNamespace(batches=SimpleNamespace(create=self.create, retrieve=self.retrieve, results=self.results))

    def create(self, requests):
        """
        Input: requests, the batch requests.
        Output: a batch with id msgbatch_1.
        What it does: records the requests.
        """
        self.state["created"].append(requests)
        return SimpleNamespace(id="msgbatch_1")

    def retrieve(self, batch_id):
        """
        Input: batch_id.
        Output: the batch, "in_progress" until checks_before_end checks have been made, then "ended".
        What it does: records the check.
        """
        self.state["retrieved"].append(batch_id)
        if len(self.state["retrieved"]) > self.state["checks_before_end"]:
            return SimpleNamespace(processing_status="ended", request_counts=SimpleNamespace(succeeded=2))
        return SimpleNamespace(processing_status="in_progress", request_counts=SimpleNamespace(succeeded=0))

    def results(self, batch_id):
        """
        Input: batch_id.
        Output: one result per entry in state["results"]: (custom_id, type, text, input tokens, output tokens), with the
        cache write and read tokens of state["cache_tokens"] (0 and 0 for an id not in it).
        What it does: builds the results the way the SDK returns them; input_tokens leaves out the cached tokens.
        """
        entries = []
        for custom_id, result_type, text, input_tokens, output_tokens in self.state["results"]:
            cache_creation_tokens, cache_read_tokens = self.state["cache_tokens"].get(custom_id, (0, 0))
            usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                                    cache_creation_input_tokens=cache_creation_tokens, cache_read_input_tokens=cache_read_tokens)
            message = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage)
            entries.append(SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=result_type, message=message)))
        return entries


@pytest.fixture
def fake_anthropic(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the state dict FakeAnthropic uses; the test sets "results" and "checks_before_end".
    What it does: makes generate_batch use FakeAnthropic. ChatAnthropic keeps the real SDK, untouched.
    """
    state = {"created": [], "retrieved": [], "checks_before_end": 0, "results": [], "cache_tokens": {}, "keys": []}

    def make_client(api_key, timeout):
        """
        Input: api_key and timeout, as anthropic.Anthropic takes them.
        Output: a FakeAnthropic.
        What it does: records the key and builds the fake client.
        """
        state["keys"].append(api_key)
        return FakeAnthropic(state)

    monkeypatch.setattr(claude_module, "anthropic", SimpleNamespace(Anthropic=make_client))
    return state


def make_claude_llm(make_test_llm_args):
    """
    Input: the make_test_llm_args fixture.
    Output: a ClaudeLLM for claude-haiku-5-5.
    What it does: builds the LLM the batch tests call.
    """
    return ClaudeLLM(make_test_llm_args(), TokenCounter())


def test_batch_sends_every_request_once_and_returns_responses(paid_env, fake_anthropic, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: two requests; the batch ends after one in_progress check.
    Output: an LLMResponse per request id with its text and tokens, the tokens added to the token counter, and the
    batch id file saved while waiting.
    What it does: checks the requests are sent with their ids, model, max_tokens and prompts, and the replies come
    back by id.
    """
    fake_anthropic["checks_before_end"] = 1
    fake_anthropic["results"] = [("hash_b", "succeeded", "Answer B.", 30, 4), ("hash_a", "succeeded", "Answer A.", 20, 3)]
    batch_id_file = str(tmp_path / "generated" / "batch_id.txt")
    claude_llm = make_claude_llm(make_test_llm_args)

    replies = claude_llm.generate_batch(batch_requests, batch_id_file, 0, 60)

    assert fake_anthropic["keys"] == ["test-claude-key"]
    sent_requests = fake_anthropic["created"][0]
    assert len(fake_anthropic["created"]) == 1
    assert sent_requests[0] == {
        "custom_id": "hash_a",
        "params": {"model": "claude-haiku-5-5", "max_tokens": 100,
                   "system": [{"type": "text", "text": "Answer from the context.", "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
                   "messages": [{"role": "user", "content": "Context: A\n\nQuestion: a?"}]},
    }
    assert sent_requests[1]["custom_id"] == "hash_b"
    assert len(fake_anthropic["retrieved"]) == 2
    assert replies["hash_a"] == LLMResponse(llm="claude", llm_model="claude-haiku-5-5", request_id="hash_a", text="Answer A.",
                                            input_tokens=20, output_tokens=3)
    assert replies["hash_b"].text == "Answer B."
    assert claude_llm.token_counter.number_of_calls == 2
    assert claude_llm.token_counter.input_tokens == 50
    assert claude_llm.token_counter.output_tokens == 7
    with open(batch_id_file, encoding="utf-8") as f:
        assert f.read() == "msgbatch_1"


def test_batch_resumes_a_saved_batch_without_sending_again(paid_env, fake_anthropic, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a batch id file from an earlier run.
    Output: the replies of that batch, with no new batch created.
    What it does: checks a stopped run resumes its batch instead of paying for the requests again.
    """
    fake_anthropic["results"] = [("hash_a", "succeeded", "Answer A.", 20, 3), ("hash_b", "succeeded", "Answer B.", 30, 4)]
    batch_id_file = tmp_path / "batch_id.txt"
    batch_id_file.write_text("msgbatch_earlier")

    replies = make_claude_llm(make_test_llm_args).generate_batch(batch_requests, str(batch_id_file), 0, 60)

    assert fake_anthropic["created"] == []
    assert fake_anthropic["retrieved"] == ["msgbatch_earlier"]
    assert sorted(replies) == ["hash_a", "hash_b"]


def test_batch_timeout_raises_and_keeps_the_batch_id(paid_env, fake_anthropic, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a batch that never ends, and a timeout of 0 seconds.
    Output: TimeoutError saying to run again, with the batch id file kept.
    What it does: checks the wait is limited and the next run can resume the same batch.
    """
    fake_anthropic["checks_before_end"] = 1000
    batch_id_file = tmp_path / "batch_id.txt"

    with pytest.raises(TimeoutError, match="Run again to resume it"):
        make_claude_llm(make_test_llm_args).generate_batch(batch_requests, str(batch_id_file), 0, 0)

    assert batch_id_file.read_text() == "msgbatch_1"


def test_batch_leaves_out_failed_requests(paid_env, fake_anthropic, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: hash_a succeeded, hash_b errored.
    Output: a reply for hash_a only, and only hash_a's tokens counted.
    What it does: checks a failed request gets no answer, so generation can send it again.
    """
    fake_anthropic["results"] = [("hash_a", "succeeded", "Answer A.", 20, 3), ("hash_b", "errored", "", 0, 0)]

    claude_llm = make_claude_llm(make_test_llm_args)
    replies = claude_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    assert list(replies) == ["hash_a"]
    assert claude_llm.token_counter.number_of_calls == 1
    assert claude_llm.token_counter.input_tokens == 20


def test_batch_counts_cache_writes_and_reads(paid_env, fake_anthropic, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: hash_a wrote 100 system prompt tokens to the cache and hash_b read them; each has its own 20 or 30 other
    input tokens.
    Output: each response's input tokens include its cached ones (120 and 130); the token counter holds 250 input
    tokens, 100 written and 100 read.
    What it does: checks Claude's input_tokens, which leaves out the cached tokens, is added up with them, so the cost
    can price each kind.
    """
    fake_anthropic["results"] = [("hash_a", "succeeded", "Answer A.", 20, 3), ("hash_b", "succeeded", "Answer B.", 30, 4)]
    fake_anthropic["cache_tokens"] = {"hash_a": (100, 0), "hash_b": (0, 100)}

    claude_llm = make_claude_llm(make_test_llm_args)
    replies = claude_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    assert (replies["hash_a"].input_tokens, replies["hash_b"].input_tokens) == (120, 130)
    assert claude_llm.token_counter.input_tokens == 250
    assert claude_llm.token_counter.cache_creation_tokens == 100
    assert claude_llm.token_counter.cache_read_tokens == 100


def test_batch_uses_the_cache_lifetime_setting(paid_env, fake_anthropic, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: cache_ttl 1h.
    Output: every request's system prompt is marked with ttl 1h.
    What it does: checks the lifetime comes from the settings, not the code.
    """
    claude_llm = ClaudeLLM(make_test_llm_args(cache_ttl="1h"), TokenCounter())

    claude_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    for sent_request in fake_anthropic["created"][0]:
        assert sent_request["params"]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
