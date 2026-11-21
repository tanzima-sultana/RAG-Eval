"""
test_query_transformation.py

Tests for rag.serving.query_transformation (query_transformation, single_query and cache). No LLM is called: the query LLM is replaced with a fake that answers
by prompt, and HuggingFaceEmbeddings with the fake from conftest. Each query type of each question is saved as its own file in tmp_path.
"""

import os
import types

import numpy as np
import pytest

from common import paths
from common.args import QueryArgs
from common.constants import I_DONT_KNOW
from common.constants import HYDE_QUERY
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import REWRITTEN_QUERY
from rag.serving.query_transformation import query_transformation as query_transformation_module
from rag.serving.query_transformation.query_transformation import QueryTransformation
from rag.serving.query_transformation.cache import load_query_entry
from rag.serving.query_transformation.cache import save_query_entry
from rag.serving.query_transformation.single_query import transform_single_query
from schema.llm import LLMArgs
from schema.query import Ans
from schema.query import Query
from schema.query import Qus

ALL_QUERY_TYPES = [ORIGINAL_QUERY, REWRITTEN_QUERY, MULTI_QUERY, HYDE_QUERY]


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
        What it does: records the model.
        """
        FakeQueryLLMProvider.built_models.append(llm_args.llm_model)

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
def fake_query_llm(monkeypatch, fake_huggingface_embeddings):
    """
    Input: pytest's monkeypatch fixture; the fake_huggingface_embeddings fixture.
    Output: the FakeQueryLLMProvider class.
    What it does: makes QueryTransformation use the fake LLM and the fake embeddings.
    """
    monkeypatch.setattr(query_transformation_module, "LLMProvider", FakeQueryLLMProvider)
    monkeypatch.setattr(query_transformation_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
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
    field_values["embedding_model"] = "fake-model"
    field_values["n_multi_qus"] = 3
    field_values["query_types"] = list(ALL_QUERY_TYPES)
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
    Output: a Query with small distinct vectors. A query type not in query_types is None.
    What it does: builds one Query to save and load.
    """
    if query_types is None:
        query_types = ALL_QUERY_TYPES
    base = float(len(query_id))

    original_qus = None
    if ORIGINAL_QUERY in query_types:
        original_qus = Qus("original", ["original"], np.array([base, 0.0, 0.0], dtype=np.float32))
    rewritten_qus = None
    if REWRITTEN_QUERY in query_types:
        rewritten_qus = Qus("rewritten", ["rewritten"], np.array([base, 1.0, 0.0], dtype=np.float32))
    multi_qus = None
    if MULTI_QUERY in query_types:
        multi_qus = []
        for multi_index in range(multi_count):
            multi_qus.append(Qus(f"multi {multi_index}", ["multi", str(multi_index)], np.array([base, 2.0, float(multi_index)], dtype=np.float32)))
    hyde_ans = None
    if HYDE_QUERY in query_types:
        hyde_ans = Ans("hyde passage", np.array([base, 3.0, 0.0], dtype=np.float32))

    return Query(
        id=query_id,
        original_qus=original_qus,
        rewritten_qus=rewritten_qus,
        multi_qus=multi_qus,
        hyde_ans=hyde_ans,
        normalized=True,
        embedding_model="fake-model",
        llm="ollama_chat",
        llm_model="llama3.1:8b",
    )


def read_file_bytes(file_path):
    """
    Input: file_path.
    Output: the file's bytes.
    What it does: reads the whole file, so a test can check it was not rewritten.
    """
    with open(file_path, "rb") as f:
        return f.read()


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"embedding_model": " "}, "embedding_model"),
    ({"n_multi_qus": 0}, "n_multi_qus"),
    ({"query_types": []}, "query_types"),
    ({"query_types": ["original", "keyword"]}, "query_type"),
])
def test_validate_raises_for_invalid_setting(overrides, message):
    """
    Input: QueryArgs with one invalid setting.
    Output: ValueError.
    What it does: checks validate names the invalid setting.
    """
    with pytest.raises(ValueError, match=message):
        QueryTransformation(make_query_args(**overrides), "out").validate()


# ---------- save_query_entry / load_query_entry ----------


@pytest.mark.parametrize("query_type", ALL_QUERY_TYPES)
def test_save_and_load_query_entry_round_trip(tmp_path, query_type):
    """
    Input: one result of each query type, saved and loaded back.
    Output: the loaded result.
    What it does: checks the texts, tokens and every vector come back.
    """
    query = make_query("q_a", 3)
    results = {
        ORIGINAL_QUERY: query.original_qus,
        REWRITTEN_QUERY: query.rewritten_qus,
        MULTI_QUERY: query.multi_qus,
        HYDE_QUERY: query.hyde_ans,
    }
    query_file = os.path.join(str(tmp_path), "queries", "entry.json")

    save_query_entry(query_file, query_type, "What is X?", results[query_type], "fake-model", "ollama_chat/llama3.1:8b")
    loaded_result = load_query_entry(query_file)

    if query_type == MULTI_QUERY:
        assert len(loaded_result) == 3
        assert loaded_result[2].qus == "multi 2"
        assert loaded_result[2].tokens == ["multi", "2"]
        assert np.array_equal(loaded_result[2].embeddings, [3.0, 2.0, 2.0])
    elif query_type == HYDE_QUERY:
        assert loaded_result.ans == "hyde passage"
        assert np.array_equal(loaded_result.embeddings, [3.0, 3.0, 0.0])
    else:
        assert loaded_result.qus == results[query_type].qus
        assert loaded_result.tokens == results[query_type].tokens
        assert np.array_equal(loaded_result.embeddings, results[query_type].embeddings)
    assert not os.path.exists(query_file + ".tmp")


def test_save_query_entry_keeps_an_empty_multi(tmp_path):
    """
    Input: a multi result with no phrasings, saved and loaded back.
    Output: the loaded result.
    What it does: checks an empty multi, the I don't know case, loads back as an empty list.
    """
    query_file = os.path.join(str(tmp_path), "entry.json")

    save_query_entry(query_file, MULTI_QUERY, "What is X?", [], "fake-model", "ollama_chat/llama3.1:8b")

    assert load_query_entry(query_file) == []


def test_load_query_entry_gives_none_when_the_file_is_missing(tmp_path):
    """
    Input: a path with no file.
    Output: None.
    What it does: checks a missing entry is reported as None.
    """
    assert load_query_entry(os.path.join(str(tmp_path), "missing.json")) is None


# ---------- get_query_file_path ----------


def get_query_file_name(query_args, query_type, question):
    """
    Input: query_args; query_type; question.
    Output: the file name of that query type of that question.
    What it does: builds the query file path under "out" and keeps only its name.
    """
    return os.path.basename(paths.get_query_file_path("out", query_args, query_type, question))


def test_get_query_file_path_is_a_12_character_hash_in_the_given_folder():
    """
    Input: the default query args, the rewritten query type and one question.
    Output: the query file path.
    What it does: checks the file is {hash}.json directly in the given folder, with a 12 hex character hash, the same
    for the same inputs.
    """
    query_file = paths.get_query_file_path("out", make_query_args(), REWRITTEN_QUERY, "What is X?")
    same_query_file = paths.get_query_file_path("out", make_query_args(), REWRITTEN_QUERY, "What is X?")

    file_name = os.path.basename(query_file)
    assert os.path.dirname(query_file) == "out"
    assert file_name.endswith(".json")
    assert len(file_name) == 12 + len(".json")
    int(file_name[:12], 16)
    assert query_file == same_query_file


@pytest.mark.parametrize("query_type", ALL_QUERY_TYPES)
def test_get_query_file_path_changes_with_the_question_query_type_and_embedding_model(query_type):
    """
    Input: one query type, and the same with another question, another query type or another embedding model.
    Output: the file names.
    What it does: checks each of these gives a different file.
    """
    default_name = get_query_file_name(make_query_args(), query_type, "What is X?")

    other_query_type = ORIGINAL_QUERY
    if query_type == ORIGINAL_QUERY:
        other_query_type = REWRITTEN_QUERY

    assert get_query_file_name(make_query_args(), query_type, "What is Y?") != default_name
    assert get_query_file_name(make_query_args(), other_query_type, "What is X?") != default_name
    assert get_query_file_name(make_query_args(embedding_model="other-model"), query_type, "What is X?") != default_name


@pytest.mark.parametrize("overrides", [
    {"query_llm_args": LLMArgs("ollama_chat/other-model", 100, 30)},
    {"query_llm_args": LLMArgs("ollama_chat/llama3.1:8b", 200, 30)},
    {"ollama_context_length": 4096},
])
def test_get_query_file_path_of_an_llm_query_type_changes_with_the_llm_settings_and_original_does_not(overrides):
    """
    Input: the default query args, and the same with another LLM model, max_tokens or Ollama context length.
    Output: the file names.
    What it does: checks the LLM query types get a different file, and original, which uses no LLM, keeps its file.
    """
    for query_type in [REWRITTEN_QUERY, MULTI_QUERY, HYDE_QUERY]:
        assert get_query_file_name(make_query_args(**overrides), query_type, "What is X?") != get_query_file_name(make_query_args(), query_type, "What is X?")
    assert get_query_file_name(make_query_args(**overrides), ORIGINAL_QUERY, "What is X?") == get_query_file_name(make_query_args(), ORIGINAL_QUERY, "What is X?")


def test_get_query_file_path_uses_n_multi_qus_only_for_multi():
    """
    Input: the default query args, and the same with another n_multi_qus.
    Output: the file names.
    What it does: checks only multi gets a different file.
    """
    changed_args = make_query_args(n_multi_qus=5)

    assert get_query_file_name(changed_args, MULTI_QUERY, "What is X?") != get_query_file_name(make_query_args(), MULTI_QUERY, "What is X?")
    for query_type in [ORIGINAL_QUERY, REWRITTEN_QUERY, HYDE_QUERY]:
        assert get_query_file_name(changed_args, query_type, "What is X?") == get_query_file_name(make_query_args(), query_type, "What is X?")


def test_get_query_file_path_ignores_the_llm_timeout_and_a_none_ollama_context_length():
    """
    Input: query args that differ only in the LLM timeout, and a Claude query LLM with no Ollama settings.
    Output: the file names.
    What it does: checks the timeout never changes the file, and a None Ollama context length is left out of the hash.
    """
    short_timeout = make_query_args(query_llm_args=LLMArgs("ollama_chat/llama3.1:8b", 100, 30))
    long_timeout = make_query_args(query_llm_args=LLMArgs("ollama_chat/llama3.1:8b", 100, 300))
    claude_args = make_query_args(query_llm_args=LLMArgs("anthropic/claude-haiku-5-5", 100, 30), ollama_api_base=None, ollama_context_length=None)

    assert get_query_file_name(short_timeout, REWRITTEN_QUERY, "What is X?") == get_query_file_name(long_timeout, REWRITTEN_QUERY, "What is X?")
    assert get_query_file_name(claude_args, REWRITTEN_QUERY, "What is X?").endswith(".json")


# ---------- transform_query ----------


def save_every_query_type(output_path, query_args, question):
    """
    Input: output_path; query_args; question.
    Output: nothing.
    What it does: saves an entry of each of the four query types of question, from make_query.
    """
    query = make_query("saved", 3)
    results = {
        ORIGINAL_QUERY: query.original_qus,
        REWRITTEN_QUERY: query.rewritten_qus,
        MULTI_QUERY: query.multi_qus,
        HYDE_QUERY: query.hyde_ans,
    }
    for query_type in ALL_QUERY_TYPES:
        query_file = paths.get_query_file_path(output_path, query_args, query_type, question)
        save_query_entry(query_file, query_type, question, results[query_type], "fake-model", "ollama_chat/llama3.1:8b")


def test_transform_query_loads_every_saved_query_type_without_building_an_llm(fake_query_llm, fake_huggingface_embeddings, tmp_path, capsys):
    """
    Input: every query type of one question saved.
    Output: the queries, the built models, what was printed and the saved files.
    What it does: checks the saved entries are returned with the question's id, no LLM or embedding is made and no
    file is rewritten.
    """
    query_args = make_query_args()
    output_path = str(tmp_path)
    save_every_query_type(output_path, query_args, "What is X?")
    rewritten_file = paths.get_query_file_path(output_path, query_args, REWRITTEN_QUERY, "What is X?")
    rewritten_bytes = read_file_bytes(rewritten_file)

    queries = QueryTransformation(query_args, output_path).transform_query(["eval_qna_0"], ["What is X?"])

    assert len(queries) == 1
    assert queries[0].id == "eval_qna_0"
    assert queries[0].rewritten_qus.qus == "rewritten"
    assert len(queries[0].multi_qus) == 3
    assert fake_query_llm.built_models == []
    assert fake_huggingface_embeddings.embed_documents_calls == []
    assert "query types loaded from cache : 4, made : 0" in capsys.readouterr().out
    assert read_file_bytes(rewritten_file) == rewritten_bytes


def test_transform_query_makes_and_saves_every_query_type_when_none_is_saved(fake_query_llm, tmp_path):
    """
    Input: one question and nothing saved.
    Output: the queries and the saved entries.
    What it does: checks every query type is made with the query LLM and saved as its own file, so the next call loads it.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "What does X mean?", "multi": "a\nb\nc", "hyde": "X is a thing."}
    query_args = make_query_args()
    output_path = str(tmp_path)

    queries = QueryTransformation(query_args, output_path).transform_query(["eval_qna_0"], ["What is X?"])

    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]
    assert fake_query_llm.reply_types == ["rewrite", "multi", "hyde"]
    assert queries[0].rewritten_qus.qus == "What does X mean?"
    assert load_query_entry(paths.get_query_file_path(output_path, query_args, ORIGINAL_QUERY, "What is X?")).qus == "What is X?"
    assert load_query_entry(paths.get_query_file_path(output_path, query_args, REWRITTEN_QUERY, "What is X?")).qus == "What does X mean?"
    assert len(load_query_entry(paths.get_query_file_path(output_path, query_args, MULTI_QUERY, "What is X?"))) == 3
    assert load_query_entry(paths.get_query_file_path(output_path, query_args, HYDE_QUERY, "What is X?")).ans == "X is a thing."


def test_transform_query_makes_only_the_missing_query_types(fake_query_llm, tmp_path):
    """
    Input: original and rewritten of one question saved; all four requested.
    Output: the queries and the prompts the LLM answered.
    What it does: checks only multi and hyde are made, and the saved rewrite is used as it is.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "A new rewrite?", "multi": "a\nb\nc", "hyde": "X is a thing."}
    query_args = make_query_args()
    output_path = str(tmp_path)
    query = make_query("saved", 0)
    save_query_entry(paths.get_query_file_path(output_path, query_args, ORIGINAL_QUERY, "What is X?"), ORIGINAL_QUERY, "What is X?", query.original_qus, "fake-model", None)
    save_query_entry(paths.get_query_file_path(output_path, query_args, REWRITTEN_QUERY, "What is X?"), REWRITTEN_QUERY, "What is X?", query.rewritten_qus, "fake-model", "ollama_chat/llama3.1:8b")

    queries = QueryTransformation(query_args, output_path).transform_query(["eval_qna_0"], ["What is X?"])

    assert fake_query_llm.reply_types == ["multi", "hyde"]
    assert queries[0].rewritten_qus.qus == "rewritten"
    assert len(queries[0].multi_qus) == 3
    assert queries[0].hyde_ans.ans == "X is a thing."


def test_transform_query_returns_only_the_requested_query_types(fake_query_llm, tmp_path):
    """
    Input: every query type of one question saved; only original and multi requested.
    Output: the queries.
    What it does: checks the other query types are None.
    """
    query_args = make_query_args(query_types=[ORIGINAL_QUERY, MULTI_QUERY])
    output_path = str(tmp_path)
    save_every_query_type(output_path, make_query_args(), "What is X?")

    queries = QueryTransformation(query_args, output_path).transform_query(["eval_qna_0"], ["What is X?"])

    assert queries[0].original_qus.qus == "original"
    assert len(queries[0].multi_qus) == 3
    assert queries[0].rewritten_qus is None
    assert queries[0].hyde_ans is None
    assert fake_query_llm.built_models == []


def test_transform_query_builds_no_llm_when_only_original_is_missing(fake_query_llm, tmp_path):
    """
    Input: nothing saved; only original requested.
    Output: the queries and the built models.
    What it does: checks the original question is embedded without building the query LLM.
    """
    query_args = make_query_args(query_types=[ORIGINAL_QUERY])

    queries = QueryTransformation(query_args, str(tmp_path)).transform_query(["eval_qna_0"], ["What is X?"])

    assert fake_query_llm.built_models == []
    assert queries[0].original_qus.qus == "What is X?"
    assert queries[0].rewritten_qus is None


def test_transform_query_keeps_the_question_order_and_ids(fake_query_llm, tmp_path):
    """
    Input: three questions, only original requested.
    Output: the queries.
    What it does: checks one Query per question, in the same order, each with its own id and question.
    """
    query_args = make_query_args(query_types=[ORIGINAL_QUERY])

    queries = QueryTransformation(query_args, str(tmp_path)).transform_query(["id_c", "id_a", "id_b"], ["Q c?", "Q a?", "Q b?"])

    query_ids = []
    query_questions = []
    for query in queries:
        query_ids.append(query.id)
        query_questions.append(query.original_qus.qus)
    assert query_ids == ["id_c", "id_a", "id_b"]
    assert query_questions == ["Q c?", "Q a?", "Q b?"]


def test_transform_query_raises_when_ids_and_questions_differ_in_length(fake_query_llm, tmp_path):
    """
    Input: two ids and one question.
    Output: ValueError.
    What it does: checks every question needs its own id.
    """
    with pytest.raises(ValueError, match="same length"):
        QueryTransformation(make_query_args(), str(tmp_path)).transform_query(["id_a", "id_b"], ["Q a?"])


# ---------- one method per query type ----------


def test_transform_single_query_original_embeds_the_question_as_it_is(fake_huggingface_embeddings):
    """
    Input: one question and query_types only original, with no LLM.
    Output: its Query.
    What it does: checks the text, BM25 tokens and vector of the question as it is, and that the other types are None.
    """
    query = transform_single_query(make_query_args(), "q1", "What is X?", [ORIGINAL_QUERY], None, fake_huggingface_embeddings())

    assert query.id == "q1"
    assert query.original_qus.qus == "What is X?"
    assert query.original_qus.tokens == ["what", "is", "x?"]
    # The fake embeds a text as [len(text), 1, 0, 0]
    assert query.original_qus.embeddings[0] == len("What is X?")
    assert query.rewritten_qus is None
    assert query.multi_qus is None
    assert query.hyde_ans is None


def test_transform_single_query_rewritten_strips_and_embeds_the_rewrite(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question with a rewrite padded by spaces.
    Output: its Query.
    What it does: checks the rewrite is stripped, split into tokens and embedded.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": " What does X mean? "}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [REWRITTEN_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.rewritten_qus.qus == "What does X mean?"
    assert query.rewritten_qus.tokens == ["what", "does", "x", "mean?"]
    assert query.rewritten_qus.embeddings[0] == len("What does X mean?")


def test_transform_single_query_multi_keeps_each_phrasing(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question with 3 phrasings and a blank line between them.
    Output: its Query.
    What it does: checks blank lines are skipped and each phrasing gets its own vector.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "Define X.\n\nExplain X.\nWhat is meant by X?"}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [MULTI_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    multi_texts = []
    for multi_qus in query.multi_qus:
        multi_texts.append(multi_qus.qus)
    assert multi_texts == ["Define X.", "Explain X.", "What is meant by X?"]
    assert query.multi_qus[1].embeddings[0] == len("Explain X.")


def test_transform_single_query_multi_keeps_no_phrasings_when_line_count_is_wrong(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question whose multi-query reply has 2 lines instead of 3.
    Output: its Query.
    What it does: checks multi holds no phrasings, so retrieval counts it as a miss.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "only one\nonly two"}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [MULTI_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.multi_qus == []


@pytest.mark.parametrize("hyde_reply", ["I don't know.", "I do not know the answer", "I don’t know"])
def test_transform_single_query_hyde_stores_refusal_as_i_dont_know(fake_query_llm, fake_huggingface_embeddings, hyde_reply):
    """
    Input: a HyDE reply that refuses.
    Output: its Query.
    What it does: checks the refusal is stored as exactly I_DONT_KNOW, so retrieval can skip it.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": hyde_reply}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [HYDE_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.hyde_ans.ans == I_DONT_KNOW


def test_transform_single_query_hyde_embeds_the_passage(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question with a HyDE passage.
    Output: its Query.
    What it does: checks the passage and its vector.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": "X is a thing used for Y."}

    query = transform_single_query(
        make_query_args(), "q1", "What is X?", [HYDE_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.hyde_ans.ans == "X is a thing used for Y."
    assert query.hyde_ans.embeddings[0] == len("X is a thing used for Y.")


def test_transform_single_query_builds_every_requested_query_type(fake_query_llm, fake_huggingface_embeddings):
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
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.id == "live_1"
    assert query.original_qus.qus == "What is X?"
    assert query.rewritten_qus.qus == "What does X mean?"
    assert len(query.multi_qus) == 3
    assert query.hyde_ans.ans == "X is a thing used for Y."
    assert query.llm == "ollama_chat"
    assert query.llm_model == "llama3.1:8b"


def test_transform_query_counts_multi_and_hyde_i_dont_know(fake_query_llm, tmp_path, capsys):
    """
    Input: two questions, the first with a bad multi reply and a HyDE refusal, and nothing saved.
    Output: the queries and what was printed.
    What it does: checks each question keeps its own phrasings and the I don't know counts are printed.
    """
    fake_query_llm.replies["Q one?"] = {"multi": "bad", "hyde": "I don't know."}
    fake_query_llm.replies["Q two?"] = {"multi": "aa\nbbbb\ncccccc", "hyde": "A passage."}
    query_args = make_query_args(query_types=[MULTI_QUERY, HYDE_QUERY])

    queries = QueryTransformation(query_args, str(tmp_path)).transform_query(["eval_qna_0", "eval_qna_1"], ["Q one?", "Q two?"])

    assert queries[0].multi_qus == []
    lengths = []
    for multi_qus in queries[1].multi_qus:
        lengths.append(multi_qus.embeddings[0])
    assert lengths == [2.0, 4.0, 6.0]
    printed = capsys.readouterr().out
    assert "Multi-query I don't know : 1" in printed
    assert "HyDE I don't know : 1" in printed


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


def test_split_queries_gives_the_same_disjoint_val_and_test_every_run():
    """
    Input: 10 queries, 6 for val and 3 for test.
    Output: the val and test queries, twice.
    What it does: checks the sizes, that the split is the same every run and that val and test do not overlap.
    """
    queries = []
    for query_number in range(10):
        queries.append(make_query(f"q_{query_number}", 0))
    query_args = make_query_args(val_queries_size=6, test_queries_size=3)

    val_queries, test_queries = QueryTransformation(query_args, "out").split_queries(queries)
    val_queries_again, test_queries_again = QueryTransformation(query_args, "out").split_queries(queries)

    val_ids = get_query_ids(val_queries)
    test_ids = get_query_ids(test_queries)
    assert len(val_ids) == 6
    assert len(test_ids) == 3
    assert val_ids == get_query_ids(val_queries_again)
    assert test_ids == get_query_ids(test_queries_again)
    assert set(val_ids).isdisjoint(test_ids)


def test_split_queries_uses_all_queries_for_both_when_split_is_off():
    """
    Input: 3 queries and query_split false, with sizes that would not fit.
    Output: the val and test queries.
    What it does: checks val and test are both all the queries, in their order, and the sizes are not used.
    """
    queries = [make_query("q_a", 0), make_query("q_b", 0), make_query("q_c", 0)]
    query_args = make_query_args(query_split=False, val_queries_size=50, test_queries_size=50)

    val_queries, test_queries = QueryTransformation(query_args, "out").split_queries(queries)

    assert get_query_ids(val_queries) == ["q_a", "q_b", "q_c"]
    assert get_query_ids(test_queries) == ["q_a", "q_b", "q_c"]


@pytest.mark.parametrize("val_queries_size, test_queries_size, message", [
    (0, 1, "at least 1"),
    (1, 0, "at least 1"),
    (2, 2, "must not be more than"),
])
def test_split_queries_raises_for_invalid_sizes(val_queries_size, test_queries_size, message):
    """
    Input: 3 queries and a val or test size that is below 1 or does not fit.
    Output: ValueError.
    What it does: checks the split never returns an empty set or reuses a query.
    """
    queries = [make_query("q_a", 0), make_query("q_b", 0), make_query("q_c", 0)]
    query_args = make_query_args(val_queries_size=val_queries_size, test_queries_size=test_queries_size)

    with pytest.raises(ValueError, match=message):
        QueryTransformation(query_args, "out").split_queries(queries)
