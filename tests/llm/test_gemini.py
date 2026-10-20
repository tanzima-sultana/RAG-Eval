"""
test_gemini.py

Tests for llm.gemini: validate_gemini and GeminiLLM.generate_batch. The list-models requests go to
fake_urlopen and the Google batch client is FakeGenaiClient, so nothing is sent.
"""

from types import SimpleNamespace

import pytest

from llm import gemini as gemini_module
from llm.gemini import GeminiLLM
from llm.gemini import validate_gemini
from llm.llm import TokenCounter
from llm.schema import LLMResponse

FIRST_PAGE = {"models": [{"name": "models/gemini-3.5-flash-lite"}], "nextPageToken": "page_2"}
SECOND_PAGE = {"models": [{"name": "models/gemini-3.1-flash-lite"}]}


def make_gemini_llm_args(make_test_llm_args, **overrides):
    """
    Input: the make_test_llm_args fixture; any LLMArgs setting as a keyword argument.
    Output: LLMArgs for gemini-3.1-flash-lite, with the overrides.
    What it does: builds the settings of the Gemini tests.
    """
    return make_test_llm_args(llm="gemini", llm_model="gemini-3.1-flash-lite", **overrides)


# ------ validate_gemini

def test_valid_gemini_settings_pass_after_reading_every_page(paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: a model listed on the second page of Google's models list.
    Output: True, after two requests, the second with the first page's token.
    What it does: checks every page of the list is read, and the "models/" prefix is removed from the names.
    """
    fake_urlopen["bodies"] = [FIRST_PAGE, SECOND_PAGE]

    assert validate_gemini(make_gemini_llm_args(make_test_llm_args)) is True
    assert len(fake_urlopen["urls"]) == 2
    assert "key=test-gemini-key" in fake_urlopen["urls"][0]
    assert "pageToken=page_2" in fake_urlopen["urls"][1]


def test_gemini_without_allow_paid_api_raises_before_any_request(no_paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: ALLOW_PAID_API not set.
    Output: ValueError mentioning ALLOW_PAID_API, and no request sent.
    What it does: checks a paid run must opt in before anything else.
    """
    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        validate_gemini(make_gemini_llm_args(make_test_llm_args))
    assert fake_urlopen["urls"] == []


def test_gemini_without_key_raises(paid_env, monkeypatch, fake_urlopen, make_test_llm_args):
    """
    Input: GEMINI_API_KEY not set.
    Output: ValueError mentioning GEMINI_API_KEY.
    What it does: checks the key is required.
    """
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        validate_gemini(make_gemini_llm_args(make_test_llm_args))


def test_gemini_model_not_listed_raises(paid_env, fake_urlopen, make_test_llm_args):
    """
    Input: gemini-0, on neither page.
    Output: ValueError saying it is not one of the Gemini models.
    What it does: checks a misspelled or retired model is caught before any paid call.
    """
    fake_urlopen["bodies"] = [FIRST_PAGE, SECOND_PAGE]
    with pytest.raises(ValueError, match="not one of the Gemini models"):
        validate_gemini(make_test_llm_args(llm="gemini", llm_model="gemini-0"))


def test_gemini_models_not_reachable_raises(paid_env, fake_urlopen, url_error, make_test_llm_args):
    """
    Input: the list-models request fails.
    Output: ValueError saying the models could not be listed.
    What it does: checks a network failure is reported, not ignored.
    """
    fake_urlopen["error"] = url_error
    with pytest.raises(ValueError, match="Could not list the Gemini models"):
        validate_gemini(make_gemini_llm_args(make_test_llm_args))


@pytest.mark.parametrize("price_name", ["input_price", "output_price", "cache_write_price", "cache_read_price"])
def test_gemini_zero_price_raises(paid_env, fake_urlopen, make_test_llm_args, price_name):
    """
    Input: one of the four prices at 0.
    Output: ValueError saying the model needs prices above 0.
    What it does: checks every paid run can compute its cost.
    """
    fake_urlopen["bodies"] = [FIRST_PAGE, SECOND_PAGE]
    with pytest.raises(ValueError, match="price above 0"):
        validate_gemini(make_gemini_llm_args(make_test_llm_args, **{price_name: 0}))


# ------ generate_batch

class FakeGenaiClient:
    """
    Stand-in for google.genai.Client: a batches client whose job ends after a set number of checks.
    """

    def __init__(self, state):
        """
        Input: state, the test's dict of settings and records.
        Output: nothing.
        What it does: builds batches with create and get.
        """
        self.state = state
        self.batches = SimpleNamespace(create=self.create, get=self.get)

    def create(self, model, src):
        """
        Input: model and src, the inline requests.
        Output: a batch job named batches/job_1.
        What it does: records the model and requests.
        """
        self.state["created"].append((model, src))
        return SimpleNamespace(name="batches/job_1")

    def get(self, name):
        """
        Input: name, the batch job name.
        Output: the job, JOB_STATE_RUNNING until checks_before_end checks have been made, then state["end_state"]
        with one inline response per entry in state["results"], whose cached_content_token_count comes from
        state["cached_tokens"] and whose thoughts_token_count comes from state["thoughts_tokens"] (None for an id not in
        them, as Gemini reports it).
        What it does: records the check.
        """
        self.state["retrieved"].append(name)
        if len(self.state["retrieved"]) <= self.state["checks_before_end"]:
            return SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_RUNNING"), dest=None)

        inlined_responses = []
        for request_id, failed, text, input_tokens, output_tokens in self.state["results"]:
            if failed:
                inlined_responses.append(SimpleNamespace(error={"code": 500}, response=None, metadata={"request_id": request_id}))
            else:
                usage_metadata = SimpleNamespace(prompt_token_count=input_tokens, candidates_token_count=output_tokens,
                                                 cached_content_token_count=self.state["cached_tokens"].get(request_id),
                                                 thoughts_token_count=self.state["thoughts_tokens"].get(request_id))
                response = SimpleNamespace(text=text, usage_metadata=usage_metadata)
                inlined_responses.append(SimpleNamespace(error=None, response=response, metadata={"request_id": request_id}))
        return SimpleNamespace(state=SimpleNamespace(name=self.state["end_state"]),
                               dest=SimpleNamespace(inlined_responses=inlined_responses))


@pytest.fixture
def fake_genai(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the state dict FakeGenaiClient uses; the test sets "results", "checks_before_end" and "end_state".
    What it does: makes generate_batch use FakeGenaiClient. ChatGoogleGenerativeAI keeps the real SDK, untouched.
    """
    state = {"created": [], "retrieved": [], "checks_before_end": 0, "end_state": "JOB_STATE_SUCCEEDED", "results": [],
             "cached_tokens": {}, "thoughts_tokens": {}, "keys": []}

    def make_client(api_key):
        """
        Input: api_key, as genai.Client takes it.
        Output: a FakeGenaiClient.
        What it does: records the key and builds the fake client.
        """
        state["keys"].append(api_key)
        return FakeGenaiClient(state)

    monkeypatch.setattr(gemini_module, "genai", SimpleNamespace(Client=make_client))
    return state


def make_gemini_llm(make_test_llm_args):
    """
    Input: the make_test_llm_args fixture.
    Output: a GeminiLLM for gemini-3.1-flash-lite.
    What it does: builds the LLM the batch tests call.
    """
    return GeminiLLM(make_gemini_llm_args(make_test_llm_args), TokenCounter())


def test_batch_sends_every_request_once_and_returns_responses(paid_env, fake_genai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: two requests; the job succeeds after one running check, with the responses in another order.
    Output: an LLMResponse per request id with its text and tokens, the tokens counted, and the job name saved.
    What it does: checks each request goes inline with its id in metadata, its system instruction and max output
    tokens, and the replies come back by that id, not by position.
    """
    fake_genai["checks_before_end"] = 1
    fake_genai["results"] = [("hash_b", False, "Answer B.", 30, 4), ("hash_a", False, "Answer A.", 20, 3)]
    batch_id_file = str(tmp_path / "generated" / "batch_id.txt")
    gemini_llm = make_gemini_llm(make_test_llm_args)

    replies = gemini_llm.generate_batch(batch_requests, batch_id_file, 0, 60)

    assert fake_genai["keys"] == ["test-gemini-key"]
    model, inline_requests = fake_genai["created"][0]
    assert model == "gemini-3.1-flash-lite"
    assert inline_requests[0] == {
        "contents": [{"role": "user", "parts": [{"text": "Context: A\n\nQuestion: a?"}]}],
        "config": {"system_instruction": "Answer from the context.", "max_output_tokens": 100},
        "metadata": {"request_id": "hash_a"},
    }
    assert len(inline_requests) == 2
    assert replies["hash_a"] == LLMResponse(llm="gemini", llm_model="gemini-3.1-flash-lite", request_id="hash_a", text="Answer A.",
                                            input_tokens=20, output_tokens=3)
    assert replies["hash_b"].text == "Answer B."
    assert gemini_llm.token_counter.number_of_calls == 2
    assert gemini_llm.token_counter.input_tokens == 50
    assert gemini_llm.token_counter.output_tokens == 7
    with open(batch_id_file, encoding="utf-8") as f:
        assert f.read() == "batches/job_1"


def test_batch_resumes_a_saved_job_without_sending_again(paid_env, fake_genai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a batch id file from an earlier run.
    Output: the replies of that job, with no new job created.
    What it does: checks a stopped run resumes its job instead of paying for the requests again.
    """
    fake_genai["results"] = [("hash_a", False, "Answer A.", 20, 3)]
    batch_id_file = tmp_path / "batch_id.txt"
    batch_id_file.write_text("batches/job_earlier")

    replies = make_gemini_llm(make_test_llm_args).generate_batch(batch_requests, str(batch_id_file), 0, 60)

    assert fake_genai["created"] == []
    assert fake_genai["retrieved"] == ["batches/job_earlier"]
    assert list(replies) == ["hash_a"]


def test_batch_timeout_raises_and_keeps_the_job_name(paid_env, fake_genai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a job that never ends, and a timeout of 0 seconds.
    Output: TimeoutError saying to run again, with the batch id file kept.
    What it does: checks the wait is limited and the next run can resume the same job.
    """
    fake_genai["checks_before_end"] = 1000
    batch_id_file = tmp_path / "batch_id.txt"

    with pytest.raises(TimeoutError, match="Run again to resume it"):
        make_gemini_llm(make_test_llm_args).generate_batch(batch_requests, str(batch_id_file), 0, 0)

    assert batch_id_file.read_text() == "batches/job_1"


def test_partly_succeeded_job_leaves_out_failed_requests(paid_env, fake_genai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: a JOB_STATE_PARTIALLY_SUCCEEDED job where hash_b failed.
    Output: a reply for hash_a only, and only its tokens counted.
    What it does: checks a partly failed job ends the wait and a failed request gets no answer, so generation can send it again.
    """
    fake_genai["end_state"] = "JOB_STATE_PARTIALLY_SUCCEEDED"
    fake_genai["results"] = [("hash_a", False, "Answer A.", 20, 3), ("hash_b", True, "", 0, 0)]

    gemini_llm = make_gemini_llm(make_test_llm_args)
    replies = gemini_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    assert list(replies) == ["hash_a"]
    assert gemini_llm.token_counter.number_of_calls == 1


def test_batch_counts_the_tokens_gemini_read_from_its_cache(paid_env, fake_genai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: hash_a's 2,000 prompt tokens include 1,200 Gemini read from its cache; hash_b reports none (None).
    Output: the token counter holds 2,100 input tokens, of which 1,200 read from the cache.
    What it does: checks Gemini's cached tokens are counted, and a missing count is taken as 0.
    """
    fake_genai["results"] = [("hash_a", False, "Answer A.", 2000, 3), ("hash_b", False, "Answer B.", 100, 4)]
    fake_genai["cached_tokens"] = {"hash_a": 1200}

    gemini_llm = make_gemini_llm(make_test_llm_args)
    gemini_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    assert gemini_llm.token_counter.input_tokens == 2100
    assert gemini_llm.token_counter.cache_read_tokens == 1200


def test_batch_counts_thinking_tokens_as_output(paid_env, fake_genai, tmp_path, make_test_llm_args, batch_requests):
    """
    Input: hash_a reports 3 answer tokens and 500 thinking tokens; hash_b reports 4 answer tokens and no thinking (None).
    Output: the token counter holds 507 output tokens, and hash_a's reply 503.
    What it does: checks a thinking model's thinking tokens, billed as output, are counted as output.
    """
    fake_genai["results"] = [("hash_a", False, "Answer A.", 2000, 3), ("hash_b", False, "Answer B.", 100, 4)]
    fake_genai["thoughts_tokens"] = {"hash_a": 500}

    gemini_llm = make_gemini_llm(make_test_llm_args)
    replies = gemini_llm.generate_batch(batch_requests, str(tmp_path / "batch_id.txt"), 0, 60)

    assert gemini_llm.token_counter.output_tokens == 507
    assert replies["hash_a"].output_tokens == 503
