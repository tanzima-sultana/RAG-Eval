"""
test_query_transformation.py

Tests for rag.serving.query (query, single_query and cache). No LLM is called: the query LLM is replaced with a fake that answers
by prompt. Every query is saved in a folder under tmp_path.
"""

import os
import types
import warnings

import pytest

from langchain_litellm import ChatLiteLLM

from common import paths
from common.args import QueryArgs
from common.args import QueryBenchmarkArgs
from common.args import QueryLLMArgs
from common.constants import I_DONT_KNOW
from common.constants import HYDE_QUERY
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import REWRITTEN_QUERY
from rag.serving.query import query as query_transformation_module
from rag.serving.query.cache import QueryLLMCache
from rag.serving.query.cache import load_query
from rag.serving.query.cache import save_query
from rag.serving.query.query import QueryTransformation
from schema.query import Query

ALL_QUERY_TYPES = [ORIGINAL_QUERY, REWRITTEN_QUERY, MULTI_QUERY, HYDE_QUERY]
LLM_QUERY_TYPES = [REWRITTEN_QUERY, MULTI_QUERY, HYDE_QUERY]


class FakeQueryLLMProvider:
    """
    Stand-in for LLMProvider. It answers each prompt type from the replies dict, keyed by the question text:
    replies[question] = {"rewrite": ..., "multi": ..., "hyde": ...}. Every built model is recorded in built_models and
    every answered prompt type in reply_types.
    """

    replies = {}
    built_models = []
    reply_types = []

    def __init__(self, llm_args, ollama_args):
        """
        Input: llm_args and ollama_args, as LLMProvider takes them.
        Output: nothing.
        What it does: records the model, and gives the fake a chat model with an empty cache field, like the
        LangChain chat model the cache is attached to.
        """
        FakeQueryLLMProvider.built_models.append(llm_args.llm_model)
        self.chat_model = types.SimpleNamespace(cache=None)

    def generate(self, system_prompt, user_prompt):
        """
        Input: system_prompt (ignored); user_prompt, one of the three query prompts.
        Output: an object with the reply in .text.
        What it does: finds the question in the prompt, records the prompt type and returns its reply.
        """
        question = user_prompt.split("Question: ", 1)[1]
        if user_prompt.startswith("Rewrite"):
            reply_type = "rewrite"
        elif user_prompt.startswith("Write exactly"):
            reply_type = "multi"
        else:
            reply_type = "hyde"
        FakeQueryLLMProvider.reply_types.append(reply_type)
        return types.SimpleNamespace(text=FakeQueryLLMProvider.replies[question][reply_type])

    def compute_cost(self):
        """
        Input: none.
        Output: nothing.
        What it does: nothing; the fake costs nothing.
        """
        return None


@pytest.fixture
def fake_query_llm(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakeQueryLLMProvider class.
    What it does: makes QueryTransformation use the fake LLM.
    """
    monkeypatch.setattr(query_transformation_module, "LLMProvider", FakeQueryLLMProvider)
    FakeQueryLLMProvider.replies = {}
    FakeQueryLLMProvider.built_models = []
    FakeQueryLLMProvider.reply_types = []
    return FakeQueryLLMProvider


def make_query_llm_args(**overrides):
    """
    Input: any QueryLLMArgs field as a keyword argument.
    Output: a QueryLLMArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["llm_model"] = "ollama_chat/llama3.1:8b"
    field_values["max_tokens"] = 100
    field_values["timeout_seconds"] = 30
    field_values["ollama_api_base"] = "http://ollama-test:11434"
    field_values["ollama_context_length"] = 32768
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryLLMArgs(**field_values)


def make_query_args(**overrides):
    """
    Input: any QueryArgs field as a keyword argument.
    Output: a QueryArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["query_llm_args"] = make_query_llm_args()
    field_values["query_type"] = REWRITTEN_QUERY
    field_values["n_multi_qus"] = 3
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryArgs(**field_values)


def make_query_benchmark_args(**overrides):
    """
    Input: any QueryBenchmarkArgs field as a keyword argument.
    Output: a QueryBenchmarkArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["query_types"] = list(ALL_QUERY_TYPES)
    field_values["query_split"] = True
    field_values["val_queries_size"] = 1
    field_values["test_queries_size"] = 1
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryBenchmarkArgs(**field_values)


def make_query_path(tmp_path):
    """
    Input: tmp_path, the test's folder.
    Output: the folder the queries are saved in.
    What it does: names a queries folder under tmp_path. The folder is created by QueryTransformation.
    """
    return str(tmp_path / "queries")


def list_query_files(query_path):
    """
    Input: query_path, the queries folder.
    Output: the names of the saved query files in it, without the LLM cache file.
    What it does: lists the .json files, so the count of saved queries can be checked.
    """
    query_file_names = []
    for file_name in os.listdir(query_path):
        if file_name.endswith(".json"):
            query_file_names.append(file_name)
    return query_file_names


def make_query(question_id):
    """
    Input: question_id.
    Output: a Query with the given id.
    What it does: builds one Query for the split tests.
    """
    return Query(
        query_type=ORIGINAL_QUERY,
        llm="ollama_chat",
        llm_model="llama3.1:8b",
        question_id=question_id,
        original_question="What is X?",
        transformed_query="What is X?",
    )


# ---------- __init__ and validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"n_multi_qus": 0}, "n_multi_qus"),
    ({"query_type": "keyword"}, "is not supported"),
    ({"query_type": "original, rewritten"}, "is not supported"),
])
def test_constructor_raises_for_invalid_setting_before_building_an_llm(fake_query_llm, tmp_path, overrides, message):
    """
    Input: QueryArgs with one invalid setting.
    Output: ValueError, and the built models.
    What it does: checks the constructor validates the settings, names the invalid one and builds no LLM.
    """
    with pytest.raises(ValueError, match=message):
        QueryTransformation(make_query_args(**overrides), make_query_path(tmp_path))

    assert fake_query_llm.built_models == []


def test_constructor_builds_no_llm_for_original(fake_query_llm, tmp_path):
    """
    Input: query_type original.
    Output: the built models and the query transformation.
    What it does: checks original is accepted and no query LLM is built.
    """
    query_transformation = QueryTransformation(make_query_args(query_type=ORIGINAL_QUERY), make_query_path(tmp_path))

    assert fake_query_llm.built_models == []
    assert query_transformation.query_llm_provider is None


@pytest.mark.parametrize("query_type", LLM_QUERY_TYPES)
def test_constructor_builds_the_llm_once_for_a_type_that_needs_it(fake_query_llm, tmp_path, query_type):
    """
    Input: query_type rewritten, multi or hyde.
    Output: the built models.
    What it does: checks the query LLM is built once, with the configured model.
    """
    QueryTransformation(make_query_args(query_type=query_type), make_query_path(tmp_path))

    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]


def test_constructor_creates_the_query_folder(fake_query_llm, tmp_path):
    """
    Input: a query folder that does not exist yet.
    Output: whether the folder exists.
    What it does: checks the constructor creates the folder the queries are saved in.
    """
    query_path = make_query_path(tmp_path)
    assert not os.path.exists(query_path)

    QueryTransformation(make_query_args(query_type=ORIGINAL_QUERY), query_path)

    assert os.path.isdir(query_path)


def test_constructor_attaches_the_llm_cache_to_the_query_llm(fake_query_llm, tmp_path):
    """
    Input: query_type rewritten, and a query folder that does not exist yet.
    Output: the cache attached to the query LLM's chat model.
    What it does: checks the folder is created and a QueryLLMCache in it, keyed with the settings LangChain leaves out,
    is attached to the query LLM.
    """
    query_path = make_query_path(tmp_path)

    query_transformation = QueryTransformation(make_query_args(query_type=REWRITTEN_QUERY), query_path)

    cache = query_transformation.query_llm_provider.chat_model.cache
    assert isinstance(cache, QueryLLMCache)
    assert cache.database_path == os.path.join(query_path, ".langchain.db")
    assert cache.settings_key == "|max_tokens=100|num_ctx=32768|api_base=http://ollama-test:11434"
    assert os.path.isdir(query_path)


def test_constructor_attaches_no_cache_for_original(fake_query_llm, tmp_path):
    """
    Input: query_type original, so no query LLM.
    Output: the query LLM and the cache file.
    What it does: checks there is no query LLM to cache and no cache file is made.
    """
    query_path = make_query_path(tmp_path)

    query_transformation = QueryTransformation(make_query_args(query_type=ORIGINAL_QUERY), query_path)

    assert query_transformation.query_llm_provider is None
    assert not os.path.exists(os.path.join(query_path, ".langchain.db"))


# ---------- QueryLLMCache ----------


def make_mocked_chat_model(cache, mock_reply, max_tokens=100, num_ctx=32768, api_base="http://localhost:11434"):
    """
    Input: cache, the QueryLLMCache; mock_reply, the reply LiteLLM gives without calling any LLM; and the settings
    the cache key adds.
    Output: a LangChain ChatLiteLLM with the cache attached.
    What it does: builds the same kind of chat model LLMProvider builds, with LiteLLM's mock_response, so no LLM is
    called.
    """
    chat_model = ChatLiteLLM(
        model="ollama_chat/llama3.1:8b",
        max_tokens=max_tokens,
        request_timeout=30,
        api_base=api_base,
        model_kwargs={"num_ctx": num_ctx, "mock_response": mock_reply},
    )
    chat_model.cache = cache
    return chat_model


def make_cache(database_path, max_tokens=100, num_ctx=32768, api_base="http://localhost:11434"):
    """
    Input: database_path; the settings the cache key adds.
    Output: a QueryLLMCache keyed with those settings, as QueryTransformation builds it.
    What it does: builds the settings_key the same way QueryTransformation does.
    """
    settings_key = f"|max_tokens={max_tokens}|num_ctx={num_ctx}|api_base={api_base}"
    return QueryLLMCache(database_path, settings_key)


def test_query_llm_cache_gives_the_saved_reply_back(tmp_path):
    """
    Input: the same prompt asked twice, through two caches on the same file, with a different mocked reply each time.
    Output: both replies.
    What it does: checks the second ask reads the first reply back as plain text, with LiteLLM's usage saved as a dict.
    """
    database_path = os.path.join(str(tmp_path), ".langchain.db")
    messages = [("user", "Rewrite the question. Question: What is X?")]

    first_reply = make_mocked_chat_model(make_cache(database_path), "reply one").invoke(messages)
    second_reply = make_mocked_chat_model(make_cache(database_path), "reply two").invoke(messages)

    assert first_reply.text == "reply one"
    assert second_reply.text == "reply one"
    assert isinstance(second_reply.response_metadata["token_usage"], dict)


@pytest.mark.parametrize("changed_setting", [
    {"max_tokens": 200},
    {"num_ctx": 4096},
    {"api_base": "http://other:11434"},
])
def test_query_llm_cache_does_not_reuse_a_reply_of_other_settings(tmp_path, changed_setting):
    """
    Input: the same prompt asked with the default settings, then with one setting LangChain leaves out of its key changed.
    Output: the second reply.
    What it does: checks the changed setting gives its own reply, not the saved one.
    """
    database_path = os.path.join(str(tmp_path), ".langchain.db")
    messages = [("user", "Rewrite the question. Question: What is X?")]
    make_mocked_chat_model(make_cache(database_path), "reply one").invoke(messages)

    second_reply = make_mocked_chat_model(make_cache(database_path, **changed_setting), "reply two", **changed_setting).invoke(messages)

    assert second_reply.text == "reply two"


def test_query_llm_cache_reads_without_a_warning(tmp_path):
    """
    Input: a reply saved, then read back.
    Output: the warnings raised while reading.
    What it does: checks reading names the allowed classes, so LangChain raises no allowed_objects warning.
    """
    database_path = os.path.join(str(tmp_path), ".langchain.db")
    messages = [("user", "Rewrite the question. Question: What is X?")]
    make_mocked_chat_model(make_cache(database_path), "reply one").invoke(messages)

    with warnings.catch_warnings(record=True) as raised_warnings:
        warnings.simplefilter("always")
        make_mocked_chat_model(make_cache(database_path), "reply two").invoke(messages)

    warning_messages = []
    for raised_warning in raised_warnings:
        warning_messages.append(str(raised_warning.message))
    for warning_message in warning_messages:
        assert "allowed_objects" not in warning_message


# ---------- query: input ----------


@pytest.mark.parametrize("question_id, question, message", [
    ("", "What is X?", "question_id is empty"),
    ("  ", "What is X?", "question_id is empty"),
    ("eval_qna_0", "", "is empty"),
    ("eval_qna_0", "   ", "is empty"),
])
def test_query_raises_for_an_empty_id_or_question(fake_query_llm, tmp_path, question_id, question, message):
    """
    Input: an empty or blank question id or question.
    Output: ValueError.
    What it does: checks the input is validated before anything is made or saved.
    """
    query_transformation = QueryTransformation(make_query_args(query_type=ORIGINAL_QUERY), make_query_path(tmp_path))

    with pytest.raises(ValueError, match=message):
        query_transformation.query(question_id, question)

    assert os.listdir(make_query_path(tmp_path)) == []


# ---------- query: original ----------


def test_query_original_keeps_the_question_and_calls_no_llm(fake_query_llm, tmp_path):
    """
    Input: one question, query_type original.
    Output: ([Query], [hash], [path]).
    What it does: checks the query holds the original question, has no transformed question and no hypothetical
    answer, carries the configured LLM, is saved at the returned path and no LLM was called.
    """
    query_path = make_query_path(tmp_path)
    query_args = make_query_args(query_type=ORIGINAL_QUERY)

    queries, query_hashes, query_file_paths = QueryTransformation(query_args, query_path).query("eval_qna_0", "What is X?")

    assert len(queries) == 1
    assert len(query_hashes) == 1
    assert len(query_file_paths) == 1
    query = queries[0]
    query_hash = query_hashes[0]
    query_file_path = query_file_paths[0]
    assert query.query_type == ORIGINAL_QUERY
    assert query.llm == "ollama_chat"
    assert query.llm_model == "llama3.1:8b"
    assert query.question_id == "eval_qna_0"
    assert query.original_question == "What is X?"
    assert query.transformed_query == "What is X?"
    assert query_hash == paths.get_query_hash(query_args, "What is X?")
    assert query_file_path == paths.get_query_file_path(query_path, query_hash)
    assert os.path.exists(query_file_path)
    assert fake_query_llm.reply_types == []


# ---------- query: rewritten ----------


def test_query_rewritten_strips_the_rewrite_and_saves_it(fake_query_llm, tmp_path):
    """
    Input: one question and an LLM reply with spaces around the rewrite.
    Output: ([Query], [hash], [path]).
    What it does: checks the transformed question is the stripped rewrite, the original question is kept, there is
    no hypothetical answer and the query is saved at the returned path.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "  What does X mean?  "}
    query_path = make_query_path(tmp_path)
    query_args = make_query_args(query_type=REWRITTEN_QUERY)

    queries, query_hashes, query_file_paths = QueryTransformation(query_args, query_path).query("eval_qna_0", "What is X?")

    assert len(queries) == 1
    query = queries[0]
    query_hash = query_hashes[0]
    query_file_path = query_file_paths[0]
    assert query.query_type == REWRITTEN_QUERY
    assert query.question_id == "eval_qna_0"
    assert query.original_question == "What is X?"
    assert query.transformed_query == "What does X mean?"
    assert query_file_path == paths.get_query_file_path(query_path, query_hash)
    assert os.path.exists(query_file_path)
    assert fake_query_llm.reply_types == ["rewrite"]


def test_query_rewritten_loads_the_saved_query_without_calling_the_llm(fake_query_llm, tmp_path):
    """
    Input: the same question sent twice, the second time to a new QueryTransformation.
    Output: the two results and the LLM's answered prompt types.
    What it does: checks the second call returns the saved query with the same hash and path, and the LLM was
    called only once.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "What does X mean?"}
    query_path = make_query_path(tmp_path)
    query_args = make_query_args(query_type=REWRITTEN_QUERY)

    first_result = QueryTransformation(query_args, query_path).query("eval_qna_0", "What is X?")
    second_result = QueryTransformation(query_args, query_path).query("eval_qna_0", "What is X?")

    assert second_result == first_result
    assert fake_query_llm.reply_types == ["rewrite"]


def test_query_rewritten_makes_a_new_query_when_a_setting_changes(fake_query_llm, tmp_path):
    """
    Input: the same question with two different max_tokens of the query LLM.
    Output: the two hashes and the LLM's answered prompt types.
    What it does: checks a changed setting gives a new hash and a new file, and the LLM is called again.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "What does X mean?"}
    query_path = make_query_path(tmp_path)
    query_args = make_query_args(query_type=REWRITTEN_QUERY)
    changed_query_args = make_query_args(query_type=REWRITTEN_QUERY, query_llm_args=make_query_llm_args(max_tokens=200))

    _, first_hashes, _ = QueryTransformation(query_args, query_path).query("eval_qna_0", "What is X?")
    _, second_hashes, _ = QueryTransformation(changed_query_args, query_path).query("eval_qna_0", "What is X?")

    assert first_hashes != second_hashes
    assert fake_query_llm.reply_types == ["rewrite", "rewrite"]
    assert len(list_query_files(query_path)) == 2


# ---------- query: hyde ----------


def test_query_hyde_keeps_the_passage_as_the_transformed_query(fake_query_llm, tmp_path, capsys):
    """
    Input: one question and an LLM reply with a passage.
    Output: ([Query], [hash], [path]) and what was printed.
    What it does: checks the passage is the hypothetical answer, the transformed question is None, the original
    question is kept and 0 refusals are printed.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": "X is a thing. It does stuff."}
    query_path = make_query_path(tmp_path)

    queries, query_hashes, query_file_paths = QueryTransformation(make_query_args(query_type=HYDE_QUERY), query_path).query("eval_qna_0", "What is X?")

    assert len(queries) == 1
    query = queries[0]
    query_file_path = query_file_paths[0]
    assert query.query_type == HYDE_QUERY
    assert query.original_question == "What is X?"
    assert query.transformed_query == "X is a thing. It does stuff."
    assert os.path.exists(query_file_path)
    assert "HyDE I don't know : 0" in capsys.readouterr().out


@pytest.mark.parametrize("hyde_reply", [
    "I don't know the answer.",
    "I do not know the answer.",
    "i don’t know.",
])
def test_query_hyde_stores_a_refusal_as_i_dont_know(fake_query_llm, tmp_path, capsys, hyde_reply):
    """
    Input: an LLM reply that refuses, in different spellings.
    Output: ([Query], [hash], [path]) and what was printed.
    What it does: checks the refusal is stored as exactly I_DONT_KNOW and 1 refusal is printed.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": hyde_reply}

    queries, _, _ = QueryTransformation(make_query_args(query_type=HYDE_QUERY), make_query_path(tmp_path)).query("eval_qna_0", "What is X?")

    assert queries[0].transformed_query == I_DONT_KNOW
    assert "HyDE I don't know : 1" in capsys.readouterr().out


# ---------- query: multi ----------


def test_query_multi_gives_one_query_hash_and_path_per_phrasing(fake_query_llm, tmp_path, capsys):
    """
    Input: one question and an LLM reply with 3 phrasings, with blank lines between them.
    Output: (list of Query, list of hashes, list of paths) and what was printed.
    What it does: checks each phrasing is its own query with the shared id and original question, each has its own
    hash and path, each file exists and 0 failures are printed.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "What does X mean?\n\n  Define X.  \nTell me about X."}
    query_path = make_query_path(tmp_path)
    query_args = make_query_args(query_type=MULTI_QUERY)

    queries, query_hashes, query_file_paths = QueryTransformation(query_args, query_path).query("eval_qna_0", "What is X?")

    transformed_queries = []
    for query in queries:
        transformed_queries.append(query.transformed_query)
        assert query.query_type == MULTI_QUERY
        assert query.question_id == "eval_qna_0"
        assert query.original_question == "What is X?"
    assert transformed_queries == ["What does X mean?", "Define X.", "Tell me about X."]
    assert len(query_hashes) == 3
    assert len(set(query_hashes)) == 3
    assert len(set(query_file_paths)) == 3
    for index in range(3):
        assert query_hashes[index] == paths.get_query_hash(query_args, transformed_queries[index])
        assert query_file_paths[index] == paths.get_query_file_path(query_path, query_hashes[index])
        assert os.path.exists(query_file_paths[index])
    assert "Multi-query I don't know : 0" in capsys.readouterr().out


@pytest.mark.parametrize("multi_reply", [
    "only one line",
    "a\nb",
    "a\nb\nc\nd",
    "",
])
def test_query_multi_keeps_nothing_when_the_line_count_is_wrong(fake_query_llm, tmp_path, capsys, multi_reply):
    """
    Input: an LLM reply with a number of lines that is not n_multi_qus (3).
    Output: the three returned lists and what was printed.
    What it does: checks no phrasing is kept or saved, the three lists are empty and 1 failure is printed.
    """
    fake_query_llm.replies["What is X?"] = {"multi": multi_reply}
    query_path = make_query_path(tmp_path)

    queries, query_hashes, query_file_paths = QueryTransformation(make_query_args(query_type=MULTI_QUERY), query_path).query("eval_qna_0", "What is X?")

    assert queries == []
    assert query_hashes == []
    assert query_file_paths == []
    assert list_query_files(query_path) == []
    assert "Multi-query I don't know : 1" in capsys.readouterr().out


def test_query_multi_calls_the_llm_every_time(fake_query_llm, tmp_path):
    """
    Input: the same question sent twice.
    Output: the LLM's answered prompt types.
    What it does: checks multi has no lookup: the LLM is called on every call, since the phrasings are not known
    before it runs.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "a\nb\nc"}
    query_transformation = QueryTransformation(make_query_args(query_type=MULTI_QUERY), make_query_path(tmp_path))

    query_transformation.query("eval_qna_0", "What is X?")
    query_transformation.query("eval_qna_0", "What is X?")

    assert fake_query_llm.reply_types == ["multi", "multi"]


# ---------- query_batch ----------


def test_query_batch_gives_a_dict_keyed_by_question_id(fake_query_llm, tmp_path):
    """
    Input: two question ids and their questions, query_type rewritten.
    Output: the dict, and the built models.
    What it does: checks each id holds the ([Query], [hash], [path]) of its own question, the LLM was built once and
    answered once per question.
    """
    fake_query_llm.replies["Q one?"] = {"rewrite": "Q one again?"}
    fake_query_llm.replies["Q two?"] = {"rewrite": "Q two again?"}
    query_transformation = QueryTransformation(make_query_args(query_type=REWRITTEN_QUERY), make_query_path(tmp_path))

    results = query_transformation.query_batch(["eval_qna_0", "eval_qna_1"], ["Q one?", "Q two?"])

    assert list(results.keys()) == ["eval_qna_0", "eval_qna_1"]
    first_queries, first_hashes, first_paths = results["eval_qna_0"]
    second_queries, second_hashes, second_paths = results["eval_qna_1"]
    assert len(first_queries) == 1
    assert len(second_queries) == 1
    assert first_queries[0].question_id == "eval_qna_0"
    assert first_queries[0].transformed_query == "Q one again?"
    assert second_queries[0].question_id == "eval_qna_1"
    assert second_queries[0].transformed_query == "Q two again?"
    assert first_hashes != second_hashes
    assert first_paths != second_paths
    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]
    assert fake_query_llm.reply_types == ["rewrite", "rewrite"]


def test_query_batch_gives_lists_for_multi(fake_query_llm, tmp_path):
    """
    Input: one question id and question, query_type multi.
    Output: the dict.
    What it does: checks the value of the id is the multi result: three queries, three hashes and three paths.
    """
    fake_query_llm.replies["Q one?"] = {"multi": "a\nb\nc"}
    query_transformation = QueryTransformation(make_query_args(query_type=MULTI_QUERY), make_query_path(tmp_path))

    results = query_transformation.query_batch(["eval_qna_0"], ["Q one?"])

    queries, query_hashes, query_file_paths = results["eval_qna_0"]
    assert len(queries) == 3
    assert len(query_hashes) == 3
    assert len(query_file_paths) == 3


def test_query_batch_raises_when_the_lists_differ_in_length(fake_query_llm, tmp_path):
    """
    Input: two ids and one question.
    Output: ValueError, and the saved files.
    What it does: checks nothing is made when the lists do not match.
    """
    query_path = make_query_path(tmp_path)
    query_transformation = QueryTransformation(make_query_args(query_type=ORIGINAL_QUERY), query_path)

    with pytest.raises(ValueError, match="must be the same length"):
        query_transformation.query_batch(["eval_qna_0", "eval_qna_1"], ["Q one?"])

    assert os.listdir(query_path) == []


# ---------- get_query_hash ----------


def test_get_query_hash_changes_with_the_text_and_the_type_and_ignores_the_timeout():
    """
    Input: QueryArgs that differ in the text, the query type, the model, n_multi_qus, or only the LLM timeout.
    Output: the hashes.
    What it does: checks every setting that shapes the query changes the hash, and the timeout does not.
    """
    query_args = make_query_args(query_type=REWRITTEN_QUERY)
    base_hash = paths.get_query_hash(query_args, "What is X?")

    assert paths.get_query_hash(query_args, "What is X?") == base_hash
    assert len(base_hash) == 12
    assert paths.get_query_hash(query_args, "What is Y?") != base_hash
    assert paths.get_query_hash(make_query_args(query_type=HYDE_QUERY), "What is X?") != base_hash
    assert paths.get_query_hash(make_query_args(query_type=REWRITTEN_QUERY, n_multi_qus=4), "What is X?") != base_hash
    different_model = make_query_args(query_type=REWRITTEN_QUERY, query_llm_args=make_query_llm_args(llm_model="ollama_chat/other:1b"))
    assert paths.get_query_hash(different_model, "What is X?") != base_hash
    different_timeout = make_query_args(query_type=REWRITTEN_QUERY, query_llm_args=make_query_llm_args(timeout_seconds=99))
    assert paths.get_query_hash(different_timeout, "What is X?") == base_hash


# ---------- save_query and load_query ----------


def test_load_query_returns_none_when_there_is_no_file(tmp_path):
    """
    Input: a file that does not exist.
    Output: the loaded value.
    What it does: checks load_query returns None.
    """
    assert load_query(str(tmp_path / "queries" / "missing.json")) is None


def test_save_query_then_load_query_gives_the_same_query_and_leaves_no_temporary_file(tmp_path):
    """
    Input: one Query with a hypothetical answer, saved into a folder that does not exist yet.
    Output: the loaded Query and the folder's files.
    What it does: checks the folder is created, the loaded query equals the saved one and no .tmp file is left.
    """
    query = Query(
        query_type=HYDE_QUERY,
        llm="ollama_chat",
        llm_model="llama3.1:8b",
        question_id="eval_qna_0",
        original_question="What is X?",
        transformed_query="X is a thing.",
    )
    query_file_path = str(tmp_path / "new_folder" / "abc.json")

    save_query(query_file_path, query)

    assert load_query(query_file_path) == query
    assert os.listdir(str(tmp_path / "new_folder")) == ["abc.json"]


# ---------- split_queries ----------


def get_query_ids(queries):
    """
    Input: queries.
    Output: their question ids, in order.
    What it does: lists the ids, so two splits can be compared.
    """
    query_ids = []
    for query in queries:
        query_ids.append(query.question_id)
    return query_ids


def test_split_queries_gives_the_same_disjoint_val_and_test_every_run(fake_query_llm, tmp_path):
    """
    Input: 10 queries, 6 for val and 3 for test.
    Output: the val and test queries, twice.
    What it does: checks the sizes, that the split is the same every run and that val and test do not overlap.
    """
    queries = []
    for query_number in range(10):
        queries.append(make_query(f"q_{query_number}"))
    query_args = make_query_args(query_type=ORIGINAL_QUERY)
    query_benchmark_args = make_query_benchmark_args(val_queries_size=6, test_queries_size=3)

    val_queries, test_queries = QueryTransformation(query_args, make_query_path(tmp_path), query_benchmark_args).split_queries(queries)
    val_queries_again, test_queries_again = QueryTransformation(query_args, make_query_path(tmp_path), query_benchmark_args).split_queries(queries)

    val_ids = get_query_ids(val_queries)
    test_ids = get_query_ids(test_queries)
    assert len(val_ids) == 6
    assert len(test_ids) == 3
    assert val_ids == get_query_ids(val_queries_again)
    assert test_ids == get_query_ids(test_queries_again)
    assert set(val_ids).isdisjoint(test_ids)


def test_split_queries_uses_all_queries_for_both_when_split_is_off(fake_query_llm, tmp_path):
    """
    Input: 3 queries and query_split false, with sizes that would not fit.
    Output: the val and test queries.
    What it does: checks val and test are both all the queries, in their order, and the sizes are not used.
    """
    queries = [make_query("q_a"), make_query("q_b"), make_query("q_c")]
    query_args = make_query_args(query_type=ORIGINAL_QUERY)
    query_benchmark_args = make_query_benchmark_args(query_split=False, val_queries_size=50, test_queries_size=50)

    val_queries, test_queries = QueryTransformation(query_args, make_query_path(tmp_path), query_benchmark_args).split_queries(queries)

    assert get_query_ids(val_queries) == ["q_a", "q_b", "q_c"]
    assert get_query_ids(test_queries) == ["q_a", "q_b", "q_c"]


@pytest.mark.parametrize("val_queries_size, test_queries_size, message", [
    (0, 1, "at least 1"),
    (1, 0, "at least 1"),
    (2, 2, "must not be more than"),
])
def test_split_queries_raises_for_invalid_sizes(fake_query_llm, tmp_path, val_queries_size, test_queries_size, message):
    """
    Input: 3 queries and a val or test size that is below 1 or does not fit.
    Output: ValueError.
    What it does: checks the split never returns an empty set or reuses a query.
    """
    queries = [make_query("q_a"), make_query("q_b"), make_query("q_c")]
    query_args = make_query_args(query_type=ORIGINAL_QUERY)
    query_benchmark_args = make_query_benchmark_args(val_queries_size=val_queries_size, test_queries_size=test_queries_size)

    with pytest.raises(ValueError, match=message):
        QueryTransformation(query_args, make_query_path(tmp_path), query_benchmark_args).split_queries(queries)


def test_split_queries_raises_when_the_layer_has_no_query_benchmark_args(fake_query_llm, tmp_path):
    """
    Input: a QueryTransformation built without query benchmark args, as the API builds it, and 3 queries.
    Output: ValueError.
    What it does: checks the layer works without the benchmark args, and only split_queries asks for them.
    """
    queries = [make_query("q_a"), make_query("q_b"), make_query("q_c")]
    query_transformation = QueryTransformation(make_query_args(query_type=ORIGINAL_QUERY), make_query_path(tmp_path))

    with pytest.raises(ValueError, match="needs query_benchmark_args"):
        query_transformation.split_queries(queries)
