"""
test_query_transformation.py

Tests for rag.serving.query (query, single_query and cache). No LLM is called: the query LLM is replaced with a fake that answers
by prompt. The query LLM's LLM cache is saved in tmp_path.
"""

import os
import types
import warnings

import pytest

from langchain_litellm import ChatLiteLLM

from common.args import QueryArgs
from common.constants import I_DONT_KNOW
from common.constants import HYDE_QUERY
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import REWRITTEN_QUERY
from rag.serving.query import query as query_transformation_module
from rag.serving.query.query import QueryTransformation
from rag.serving.query.cache import QueryLLMCache
from rag.serving.query.single_query import transform_single_query
from schema.llm import LLMArgs
from schema.query import Ans
from schema.query import Query
from schema.query import Qus

ALL_QUERY_TYPES = [ORIGINAL_QUERY, REWRITTEN_QUERY, MULTI_QUERY, HYDE_QUERY]
# The query types that can be listed in query_types. Original is always made, so it is never listed
EXTRA_QUERY_TYPES = [REWRITTEN_QUERY, MULTI_QUERY, HYDE_QUERY]


class FakeQueryLLMProvider:
    """
    Stand-in for LLMProvider. It answers each prompt type from the replies dict, keyed by the question text:
    replies[qus] = {"rewrite": ..., "multi": ..., "hyde": ...}. Every built model is recorded in built_models and
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


def make_query_args(**overrides):
    """
    Input: any QueryArgs field as a keyword argument.
    Output: a QueryArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["query_llm_args"] = LLMArgs("ollama_chat/llama3.1:8b", 100, 30)
    field_values["n_multi_qus"] = 3
    field_values["query_types"] = list(EXTRA_QUERY_TYPES)
    field_values["query_split"] = True
    field_values["val_queries_size"] = 1
    field_values["test_queries_size"] = 1
    field_values["ollama_api_base"] = "http://ollama-test:11434"
    field_values["ollama_context_length"] = 32768
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryArgs(**field_values)


def make_query(query_id, multi_count, query_types=None):
    """
    Input: query_id; multi_count, the number of multi-query phrasings; query_types, the query types the query holds
    (all four when None).
    Output: a Query of texts. A query type not in query_types is None.
    What it does: builds one Query for the split tests.
    """
    if query_types is None:
        query_types = ALL_QUERY_TYPES

    original_qus = None
    if ORIGINAL_QUERY in query_types:
        original_qus = Qus("original")
    rewritten_qus = None
    if REWRITTEN_QUERY in query_types:
        rewritten_qus = Qus("rewritten")
    multi_qus = None
    if MULTI_QUERY in query_types:
        multi_qus = []
        for multi_index in range(multi_count):
            multi_qus.append(Qus(f"multi {multi_index}"))
    hyde_ans = None
    if HYDE_QUERY in query_types:
        hyde_ans = Ans("hyde passage")

    return Query(
        id=query_id,
        original_qus=original_qus,
        rewritten_qus=rewritten_qus,
        multi_qus=multi_qus,
        hyde_ans=hyde_ans,
        llm="ollama_chat",
        llm_model="llama3.1:8b",
    )


# ---------- __init__ and validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"n_multi_qus": 0}, "n_multi_qus"),
    ({"query_types": ["keyword"]}, "is not supported"),
    ({"query_types": ["original"]}, "must not be in query_types"),
    ({"query_types": ["rewritten", "original"]}, "must not be in query_types"),
])
def test_constructor_raises_for_invalid_setting_before_building_an_llm(fake_query_llm, tmp_path, overrides, message):
    """
    Input: QueryArgs with one invalid setting.
    Output: ValueError, and the built models.
    What it does: checks the constructor validates the settings, names the invalid one and builds no LLM.
    """
    with pytest.raises(ValueError, match=message):
        QueryTransformation(make_query_args(**overrides), str(tmp_path))

    assert fake_query_llm.built_models == []


def test_constructor_builds_no_llm_when_query_types_is_empty(fake_query_llm, tmp_path):
    """
    Input: empty query_types, so original only.
    Output: the built models and the query transformation.
    What it does: checks no query LLM is built.
    """
    query_transformation = QueryTransformation(make_query_args(query_types=[]), str(tmp_path))

    assert fake_query_llm.built_models == []
    assert query_transformation.query_llm_provider is None


def test_constructor_builds_the_llm_once_for_every_question(fake_query_llm, tmp_path):
    """
    Input: every extra query type requested, nothing saved, two questions sent one at a time.
    Output: the built models.
    What it does: checks the query LLM is built once in the constructor and reused by each question.
    """
    fake_query_llm.replies["Q one?"] = {"rewrite": "Q one again?", "multi": "a\nb\nc", "hyde": "One."}
    fake_query_llm.replies["Q two?"] = {"rewrite": "Q two again?", "multi": "a\nb\nc", "hyde": "Two."}
    query_transformation = QueryTransformation(make_query_args(), str(tmp_path))

    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]

    query_transformation.query("eval_qna_0", "Q one?")
    query_transformation.query("eval_qna_1", "Q two?")

    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]


def test_constructor_attaches_the_llm_cache_to_the_query_llm(fake_query_llm, tmp_path):
    """
    Input: every extra query type requested, and a query folder that does not exist yet.
    Output: the cache attached to the query LLM's chat model.
    What it does: checks the folder is created and a QueryLLMCache in it, keyed with the settings LangChain leaves out,
    is attached to the query LLM.
    """
    query_path = os.path.join(str(tmp_path), "queries")

    query_transformation = QueryTransformation(make_query_args(), query_path)

    cache = query_transformation.query_llm_provider.chat_model.cache
    assert isinstance(cache, QueryLLMCache)
    assert cache.database_path == os.path.join(query_path, ".langchain.db")
    assert cache.settings_key == "|max_tokens=100|num_ctx=32768|api_base=http://ollama-test:11434"
    assert os.path.isdir(query_path)


def test_constructor_attaches_no_cache_when_query_types_is_empty(fake_query_llm, tmp_path):
    """
    Input: empty query_types, so no query LLM.
    Output: the query LLM and the query folder.
    What it does: checks there is no query LLM to cache and no cache file folder is made.
    """
    query_path = os.path.join(str(tmp_path), "queries")

    query_transformation = QueryTransformation(make_query_args(query_types=[]), query_path)

    assert query_transformation.query_llm_provider is None
    assert not os.path.exists(query_path)


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


# ---------- query ----------


def test_query_makes_every_query_type(fake_query_llm, tmp_path):
    """
    Input: one question, every extra query type requested.
    Output: the query and the prompts the LLM answered.
    What it does: checks original is made and each extra query type is made with the query LLM.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "What does X mean?", "multi": "a\nb\nc", "hyde": "X is a thing."}

    query = QueryTransformation(make_query_args(), str(tmp_path)).query("eval_qna_0", "What is X?")

    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]
    assert fake_query_llm.reply_types == ["rewrite", "multi", "hyde"]
    assert query.id == "eval_qna_0"
    assert query.original_qus.qus == "What is X?"
    assert query.rewritten_qus.qus == "What does X mean?"
    assert len(query.multi_qus) == 3
    assert query.hyde_ans.ans == "X is a thing."


def test_query_returns_only_the_requested_query_types(fake_query_llm, tmp_path):
    """
    Input: one question; only multi requested as an extra query type.
    Output: the query and the prompts the LLM answered.
    What it does: checks original and multi are returned, the other query types are None and only multi is asked.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "a\nb\nc"}
    query_args = make_query_args(query_types=[MULTI_QUERY])

    query = QueryTransformation(query_args, str(tmp_path)).query("eval_qna_0", "What is X?")

    assert query.original_qus.qus == "What is X?"
    assert len(query.multi_qus) == 3
    assert query.rewritten_qus is None
    assert query.hyde_ans is None
    assert fake_query_llm.reply_types == ["multi"]


def test_query_with_empty_query_types_gives_only_original_without_an_llm(fake_query_llm, tmp_path):
    """
    Input: empty query_types.
    Output: the query and the built models.
    What it does: checks the original question is embedded, every extra query type is None and no LLM is built.
    """
    query_args = make_query_args(query_types=[])

    query = QueryTransformation(query_args, str(tmp_path)).query("eval_qna_0", "What is X?")

    assert query.id == "eval_qna_0"
    assert query.original_qus.qus == "What is X?"
    assert query.rewritten_qus is None
    assert query.multi_qus is None
    assert query.hyde_ans is None
    assert fake_query_llm.built_models == []


def test_query_always_makes_original_besides_the_extra_query_types(fake_query_llm, tmp_path):
    """
    Input: only hyde requested as an extra query type.
    Output: the query.
    What it does: checks original is made even though it is not in query_types, next to hyde.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": "X is a thing."}
    query_args = make_query_args(query_types=[HYDE_QUERY])
    output_path = str(tmp_path)

    query = QueryTransformation(query_args, output_path).query("eval_qna_0", "What is X?")

    assert query.original_qus.qus == "What is X?"
    assert query.hyde_ans.ans == "X is a thing."
    assert query.rewritten_qus is None
    assert query.multi_qus is None
    assert fake_query_llm.reply_types == ["hyde"]


@pytest.mark.parametrize("question_id, question, message", [
    ("", "What is X?", "question_id"),
    ("   ", "What is X?", "question_id"),
    ("eval_qna_0", "", "question of eval_qna_0"),
    ("eval_qna_0", "  \n ", "question of eval_qna_0"),
])
def test_query_raises_for_an_empty_question_id_or_question(fake_query_llm, tmp_path, question_id, question, message):
    """
    Input: an empty or whitespace-only question id or question.
    Output: ValueError.
    What it does: checks the input is rejected and names what is empty.
    """
    query_transformation = QueryTransformation(make_query_args(query_types=[]), str(tmp_path))

    with pytest.raises(ValueError, match=message):
        query_transformation.query(question_id, question)


# ---------- one method per query type ----------


def test_transform_single_query_original_keeps_the_question_as_it_is():
    """
    Input: one question and query_types only original, with no LLM.
    Output: its Query.
    What it does: checks the text of the question as it is, and that the other types are None.
    """
    query = transform_single_query(make_query_args(), "q1", "What is X?", [ORIGINAL_QUERY], None)

    assert query.id == "q1"
    assert query.original_qus.qus == "What is X?"
    assert query.rewritten_qus is None
    assert query.multi_qus is None
    assert query.hyde_ans is None


def test_transform_single_query_rewritten_strips_the_rewrite(fake_query_llm):
    """
    Input: one question with a rewrite padded by spaces.
    Output: its Query.
    What it does: checks the rewrite is stripped.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": " What does X mean? "}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [REWRITTEN_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None)
    )

    assert query.rewritten_qus.qus == "What does X mean?"


def test_transform_single_query_multi_keeps_each_phrasing(fake_query_llm):
    """
    Input: one question with 3 phrasings and a blank line between them.
    Output: its Query.
    What it does: checks blank lines are skipped and each phrasing is kept as its own text.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "Define X.\n\nExplain X.\nWhat is meant by X?"}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [MULTI_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None)
    )

    multi_texts = []
    for multi_qus in query.multi_qus:
        multi_texts.append(multi_qus.qus)
    assert multi_texts == ["Define X.", "Explain X.", "What is meant by X?"]


def test_transform_single_query_multi_keeps_no_phrasings_when_line_count_is_wrong(fake_query_llm):
    """
    Input: one question whose multi-query reply has 2 lines instead of 3.
    Output: its Query.
    What it does: checks multi holds no phrasings, so retrieval counts it as a miss.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "only one\nonly two"}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [MULTI_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None)
    )

    assert query.multi_qus == []


@pytest.mark.parametrize("hyde_reply", ["I don't know.", "I do not know the answer", "I don’t know"])
def test_transform_single_query_hyde_stores_refusal_as_i_dont_know(fake_query_llm, hyde_reply):
    """
    Input: a HyDE reply that refuses.
    Output: its Query.
    What it does: checks the refusal is stored as exactly I_DONT_KNOW, so retrieval can skip it.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": hyde_reply}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [HYDE_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None)
    )

    assert query.hyde_ans.ans == I_DONT_KNOW


def test_transform_single_query_hyde_keeps_the_passage(fake_query_llm):
    """
    Input: one question with a HyDE passage.
    Output: its Query.
    What it does: checks the passage text.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": "X is a thing used for Y."}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [HYDE_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None)
    )

    assert query.hyde_ans.ans == "X is a thing used for Y."


def test_transform_single_query_builds_every_requested_query_type(fake_query_llm):
    """
    Input: one live question with replies for rewrite, multi and HyDE.
    Output: its Query.
    What it does: checks every query type is filled and the LLM fields come from the query model.
    """
    fake_query_llm.replies["What is X?"] = {
        "rewrite": "What does X mean?",
        "multi": "Define X.\nExplain X.\nWhat is meant by X?",
        "hyde": "X is a thing used for Y.",
    }

    query = transform_single_query(
        make_query_args(), "live_1", "What is X?", list(ALL_QUERY_TYPES),
        FakeQueryLLMProvider(make_query_args().query_llm_args, None)
    )

    assert query.id == "live_1"
    assert query.original_qus.qus == "What is X?"
    assert query.rewritten_qus.qus == "What does X mean?"
    assert len(query.multi_qus) == 3
    assert query.hyde_ans.ans == "X is a thing used for Y."
    assert query.llm == "ollama_chat"
    assert query.llm_model == "llama3.1:8b"


def test_query_counts_multi_and_hyde_i_dont_know_per_question(fake_query_llm, tmp_path, capsys):
    """
    Input: two questions sent one at a time, the first with a bad multi reply and a HyDE refusal, and nothing saved.
    Output: each query and what was printed for each question.
    What it does: checks each question keeps its own phrasings and the I don't know counts are printed per question.
    """
    fake_query_llm.replies["Q one?"] = {"multi": "bad", "hyde": "I don't know."}
    fake_query_llm.replies["Q two?"] = {"multi": "aa\nbbbb\ncccccc", "hyde": "A passage."}
    query_args = make_query_args(query_types=[MULTI_QUERY, HYDE_QUERY])
    query_transformation = QueryTransformation(query_args, str(tmp_path))

    first_query = query_transformation.query("eval_qna_0", "Q one?")
    first_printed = capsys.readouterr().out
    second_query = query_transformation.query("eval_qna_1", "Q two?")
    second_printed = capsys.readouterr().out

    assert first_query.multi_qus == []
    multi_texts = []
    for multi_qus in second_query.multi_qus:
        multi_texts.append(multi_qus.qus)
    assert multi_texts == ["aa", "bbbb", "cccccc"]
    assert "Multi-query I don't know : 1" in first_printed
    assert "HyDE I don't know : 1" in first_printed
    assert "Multi-query I don't know : 0" in second_printed
    assert "HyDE I don't know : 0" in second_printed


# ---------- split_queries ----------


def get_query_ids(queries):
    """
    Input: queries.
    Output: their ids, in order.
    What it does: lists the ids, so two splits can be compared.
    """
    query_ids = []
    for query in queries:
        query_ids.append(query.id)
    return query_ids


def test_split_queries_gives_the_same_disjoint_val_and_test_every_run(fake_query_llm, tmp_path):
    """
    Input: 10 queries, 6 for val and 3 for test.
    Output: the val and test queries, twice.
    What it does: checks the sizes, that the split is the same every run and that val and test do not overlap.
    """
    queries = []
    for query_number in range(10):
        queries.append(make_query(f"q_{query_number}", 0))
    query_args = make_query_args(val_queries_size=6, test_queries_size=3)

    val_queries, test_queries = QueryTransformation(query_args, str(tmp_path)).split_queries(queries)
    val_queries_again, test_queries_again = QueryTransformation(query_args, str(tmp_path)).split_queries(queries)

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
    queries = [make_query("q_a", 0), make_query("q_b", 0), make_query("q_c", 0)]
    query_args = make_query_args(query_split=False, val_queries_size=50, test_queries_size=50)

    val_queries, test_queries = QueryTransformation(query_args, str(tmp_path)).split_queries(queries)

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
    queries = [make_query("q_a", 0), make_query("q_b", 0), make_query("q_c", 0)]
    query_args = make_query_args(val_queries_size=val_queries_size, test_queries_size=test_queries_size)

    with pytest.raises(ValueError, match=message):
        QueryTransformation(query_args, str(tmp_path)).split_queries(queries)
