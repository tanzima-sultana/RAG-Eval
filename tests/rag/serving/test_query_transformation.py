"""
test_query_transformation.py

Tests for rag.serving.query_transformation (query_transformation, single_query and cache). No LLM is called: the query LLM is replaced with a fake that answers
by prompt, and HuggingFaceEmbeddings with the fake from conftest. The saved 200 queries are only read, never written.
"""

import json
import os
import types

import numpy as np
import pytest

from common import paths
from common.args import OllamaArgs
from common.args import EvalDatasetArgs
from common.args import QueryArgs
from common.constants import I_DONT_KNOW
from common.constants import HYDE_QUERY
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import REWRITTEN_QUERY
from rag.serving.query_transformation import query_transformation as query_transformation_module
from rag.serving.query_transformation.query_transformation import QueryTransformation
from rag.serving.query_transformation.cache import load_queries
from rag.serving.query_transformation.cache import save_queries
from rag.serving.query_transformation.single_query import transform_question
from schema.llm import LLMArgs
from schema.eval_sample import EvalSample
from schema.query import Ans
from schema.query import Query
from schema.query import Qus

SAVED_QUERY_PATH = (
    "temp/pdfa-eng-wds/queries/"
    "100_200_anthropic-claude-sonnet-5-5_all-MiniLM-L6-v2_all-MiniLM-L6-v2_ollama_chat-llama3.1-8b_3"
)


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
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryArgs(**field_values)


def make_eval_dataset_args():
    """
    Input: none.
    Output: an EvalDatasetArgs with test values.
    What it does: builds the eval dataset settings the query path is named by.
    """
    return EvalDatasetArgs(
        "outputs/",
        "test-dataset",
        1,
        LLMArgs("anthropic/claude-sonnet-5-5", 100, 30),
        LLMArgs("anthropic/claude-haiku-5-5", 100, 30),
        "fake-model",
        1,
        2,
    )


def make_ollama_args():
    """
    Input: none.
    Output: an OllamaArgs with test values.
    What it does: builds the Ollama settings QueryTransformation takes.
    """
    return OllamaArgs(ollama_api_base="http://ollama-test:11434", ollama_context_length=32768)


def make_eval_sample(item_number, question):
    """
    Input: item_number; question, the question text.
    Output: an EvalSample with id "eval_qna_{item_number}" and that question.
    What it does: builds one eval sample with placeholder values.
    """
    return EvalSample(
        id=f"eval_qna_{item_number}",
        qus=question,
        ans="An answer.",
        context=["A context."],
        qus_type="single_hop_specific_query_synthesizer",
        doc_ids=["doc1"],
        doc_paths=["data/doc1.pdf"],
        llm="anthropic",
        llm_model="claude-sonnet-5-5",
    )


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
        QueryTransformation(make_query_args(**overrides), make_ollama_args()).validate()


# ---------- save_queries / load_queries ----------


def test_save_and_load_queries_round_trip(tmp_path):
    """
    Input: one query with 3 phrasings and one with none, saved and loaded back.
    Output: the loaded queries.
    What it does: checks texts, tokens and every vector come back, including the flat multi-query rows.
    """
    queries = [make_query("q_a", 3), make_query("q_bb", 0)]

    save_queries(str(tmp_path), queries)
    loaded_queries = load_queries(str(tmp_path))

    assert len(loaded_queries) == 2
    assert loaded_queries[0].rewritten_qus.tokens == ["rewritten"]
    assert len(loaded_queries[0].multi_qus) == 3
    assert np.array_equal(loaded_queries[0].multi_qus[2].embeddings, [3.0, 2.0, 2.0])
    assert loaded_queries[1].multi_qus == []
    assert np.array_equal(loaded_queries[1].hyde_ans.embeddings, [4.0, 3.0, 0.0])
    assert load_queries(str(tmp_path / "missing")) is None


def test_save_and_load_queries_keeps_missing_query_types_as_none(tmp_path):
    """
    Input: queries holding only original and hyde, saved and loaded back.
    Output: the loaded queries and the saved vector arrays.
    What it does: checks a query type that is None is left out of both files and loads back as None.
    """
    queries = [make_query("q_a", 3, [ORIGINAL_QUERY, HYDE_QUERY])]

    save_queries(str(tmp_path), queries)
    loaded_queries = load_queries(str(tmp_path))

    assert loaded_queries[0].original_qus.qus == "original"
    assert loaded_queries[0].rewritten_qus is None
    assert loaded_queries[0].multi_qus is None
    assert loaded_queries[0].hyde_ans.ans == "hyde passage"
    with np.load(os.path.join(str(tmp_path), "embeddings.npz")) as saved_arrays:
        assert sorted(saved_arrays.files) == ["hyde", "original", "query_ids"]


def test_load_queries_raises_when_files_are_out_of_order(tmp_path):
    """
    Input: saved queries whose queries.json order no longer matches embeddings.npz.
    Output: ValueError.
    What it does: checks texts are never joined with another query's vectors.
    """
    save_queries(str(tmp_path), [make_query("q_a", 0), make_query("q_b", 0)])
    queries_file = os.path.join(str(tmp_path), "queries.json")
    with open(queries_file, encoding="utf-8") as f:
        query_dicts = json.load(f)
    query_dicts.reverse()
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(query_dicts, f)

    with pytest.raises(ValueError, match="out of order"):
        load_queries(str(tmp_path))


# ---------- compute_queries ----------


def test_compute_queries_returns_saved_queries_without_building_an_llm(fake_query_llm, fake_huggingface_embeddings, tmp_path, capsys):
    """
    Input: saved queries holding all four query types at the query path.
    Output: the queries, the built models, what was printed and the saved files.
    What it does: checks the saved queries are returned, no LLM is built and the saved files are not rewritten.
    """
    query_args = make_query_args()
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    query_path = paths.get_query_path(output_path, eval_dataset_args, query_args)
    save_queries(query_path, [make_query("eval_qna_0", 3)])
    queries_bytes = read_file_bytes(os.path.join(query_path, "queries.json"))
    embeddings_bytes = read_file_bytes(os.path.join(query_path, "embeddings.npz"))

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries(
        [make_eval_sample(0, "What is X?")], eval_dataset_args, output_path
    )

    assert len(queries) == 1
    assert fake_query_llm.built_models == []
    assert fake_huggingface_embeddings.embed_documents_calls == []
    assert "Queries : 1 (all requested query types from cache)" in capsys.readouterr().out
    assert read_file_bytes(os.path.join(query_path, "queries.json")) == queries_bytes
    assert read_file_bytes(os.path.join(query_path, "embeddings.npz")) == embeddings_bytes


def test_compute_queries_returns_only_the_requested_query_types(fake_query_llm, tmp_path):
    """
    Input: saved queries holding all four query types; only original and multi requested.
    Output: the queries and the queries saved after the call.
    What it does: checks the other query types are None in the result, but stay whole in the saved queries.
    """
    query_args = make_query_args(query_types=[ORIGINAL_QUERY, MULTI_QUERY])
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    query_path = paths.get_query_path(output_path, eval_dataset_args, query_args)
    save_queries(query_path, [make_query("eval_qna_0", 3)])

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries(
        [make_eval_sample(0, "What is X?")], eval_dataset_args, output_path
    )

    assert queries[0].original_qus.qus == "original"
    assert len(queries[0].multi_qus) == 3
    assert queries[0].rewritten_qus is None
    assert queries[0].hyde_ans is None
    assert fake_query_llm.built_models == []
    saved_queries = load_queries(query_path)
    assert saved_queries[0].rewritten_qus.qus == "rewritten"
    assert saved_queries[0].hyde_ans.ans == "hyde passage"


def test_compute_queries_generates_and_saves_when_none_are_saved(fake_query_llm, tmp_path):
    """
    Input: one question and no saved queries.
    Output: the queries and the query folder.
    What it does: checks every query type is generated with the query LLM and saved, so the next run loads them.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "What does X mean?", "multi": "a\nb\nc", "hyde": "X is a thing."}
    query_args = make_query_args()
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries([make_eval_sample(0, "What is X?")], eval_dataset_args, output_path)

    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]
    assert fake_query_llm.reply_types == ["rewrite", "multi", "hyde"]
    saved_queries = load_queries(paths.get_query_path(output_path, eval_dataset_args, query_args))
    assert saved_queries[0].original_qus.qus == "What is X?"
    assert saved_queries[0].rewritten_qus.qus == queries[0].rewritten_qus.qus
    assert len(saved_queries[0].multi_qus) == 3
    assert saved_queries[0].hyde_ans.ans == "X is a thing."


def test_compute_queries_generates_only_the_missing_query_types(fake_query_llm, tmp_path):
    """
    Input: saved queries holding original and rewritten; all four requested.
    Output: the queries, the prompts the LLM answered and the saved queries.
    What it does: checks only multi and hyde are generated, the saved rewrite is kept, and all four are saved.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "A new rewrite?", "multi": "a\nb\nc", "hyde": "X is a thing."}
    query_args = make_query_args()
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    query_path = paths.get_query_path(output_path, eval_dataset_args, query_args)
    save_queries(query_path, [make_query("eval_qna_0", 0, [ORIGINAL_QUERY, REWRITTEN_QUERY])])

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries(
        [make_eval_sample(0, "What is X?")], eval_dataset_args, output_path
    )

    assert fake_query_llm.reply_types == ["multi", "hyde"]
    assert queries[0].rewritten_qus.qus == "rewritten"
    saved_queries = load_queries(query_path)
    assert saved_queries[0].original_qus.qus == "original"
    assert saved_queries[0].rewritten_qus.qus == "rewritten"
    assert len(saved_queries[0].multi_qus) == 3
    assert saved_queries[0].hyde_ans.ans == "X is a thing."


def test_compute_queries_builds_no_llm_when_only_original_is_missing(fake_query_llm, tmp_path):
    """
    Input: no saved queries; only original requested.
    Output: the queries and the built models.
    What it does: checks the original question is embedded without building the query LLM.
    """
    query_args = make_query_args(query_types=[ORIGINAL_QUERY])
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries(
        [make_eval_sample(0, "What is X?")], eval_dataset_args, output_path
    )

    assert fake_query_llm.built_models == []
    assert queries[0].original_qus.qus == "What is X?"
    assert queries[0].rewritten_qus is None
    assert load_queries(paths.get_query_path(output_path, eval_dataset_args, query_args))[0].multi_qus is None


def test_compute_queries_raises_when_saved_queries_are_not_in_eval_dataset_order(fake_query_llm, tmp_path):
    """
    Input: saved queries whose ids are not the eval dataset ids.
    Output: ValueError.
    What it does: checks saved queries are never joined with another question.
    """
    query_args = make_query_args()
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    save_queries(paths.get_query_path(output_path, eval_dataset_args, query_args), [make_query("eval_qna_9", 3)])

    with pytest.raises(ValueError, match="not in the eval dataset order"):
        QueryTransformation(query_args, make_ollama_args()).compute_queries(
            [make_eval_sample(0, "What is X?")], eval_dataset_args, output_path
        )


# ---------- one method per query type ----------


def test_transform_question_original_embeds_the_question_as_it_is(fake_huggingface_embeddings):
    """
    Input: one question and query_types only original, with no LLM.
    Output: its Query.
    What it does: checks the text, BM25 tokens and vector of the question as it is, and that the other types are None.
    """
    query = transform_question(make_query_args(), "q1", "What is X?", [ORIGINAL_QUERY], None, fake_huggingface_embeddings())

    assert query.id == "q1"
    assert query.original_qus.qus == "What is X?"
    assert query.original_qus.tokens == ["what", "is", "x?"]
    # The fake embeds a text as [len(text), 1, 0, 0]
    assert query.original_qus.embeddings[0] == len("What is X?")
    assert query.rewritten_qus is None
    assert query.multi_qus is None
    assert query.hyde_ans is None


def test_transform_question_rewritten_strips_and_embeds_the_rewrite(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question with a rewrite padded by spaces.
    Output: its Query.
    What it does: checks the rewrite is stripped, split into tokens and embedded.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": " What does X mean? "}

    query = transform_question(
        make_query_args(), "q1", "What is X?", [REWRITTEN_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.rewritten_qus.qus == "What does X mean?"
    assert query.rewritten_qus.tokens == ["what", "does", "x", "mean?"]
    assert query.rewritten_qus.embeddings[0] == len("What does X mean?")


def test_transform_question_multi_keeps_each_phrasing(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question with 3 phrasings and a blank line between them.
    Output: its Query.
    What it does: checks blank lines are skipped and each phrasing gets its own vector.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "Define X.\n\nExplain X.\nWhat is meant by X?"}

    query = transform_question(
        make_query_args(), "q1", "What is X?", [MULTI_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    multi_texts = []
    for multi_qus in query.multi_qus:
        multi_texts.append(multi_qus.qus)
    assert multi_texts == ["Define X.", "Explain X.", "What is meant by X?"]
    assert query.multi_qus[1].embeddings[0] == len("Explain X.")


def test_transform_question_multi_keeps_no_phrasings_when_line_count_is_wrong(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question whose multi-query reply has 2 lines instead of 3.
    Output: its Query.
    What it does: checks multi holds no phrasings, so retrieval counts it as a miss.
    """
    fake_query_llm.replies["What is X?"] = {"multi": "only one\nonly two"}

    query = transform_question(
        make_query_args(), "q1", "What is X?", [MULTI_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.multi_qus == []


@pytest.mark.parametrize("hyde_reply", ["I don't know.", "I do not know the answer", "I don’t know"])
def test_transform_question_hyde_stores_refusal_as_i_dont_know(fake_query_llm, fake_huggingface_embeddings, hyde_reply):
    """
    Input: a HyDE reply that refuses.
    Output: its Query.
    What it does: checks the refusal is stored as exactly I_DONT_KNOW, so retrieval can skip it.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": hyde_reply}

    query = transform_question(
        make_query_args(), "q1", "What is X?", [HYDE_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.hyde_ans.ans == I_DONT_KNOW


def test_transform_question_hyde_embeds_the_passage(fake_query_llm, fake_huggingface_embeddings):
    """
    Input: one question with a HyDE passage.
    Output: its Query.
    What it does: checks the passage and its vector.
    """
    fake_query_llm.replies["What is X?"] = {"hyde": "X is a thing used for Y."}

    query = transform_question(
        make_query_args(), "q1", "What is X?", [HYDE_QUERY],
        FakeQueryLLMProvider(make_query_args().query_llm_args, None), fake_huggingface_embeddings()
    )

    assert query.hyde_ans.ans == "X is a thing used for Y."
    assert query.hyde_ans.embeddings[0] == len("X is a thing used for Y.")


def test_transform_question_builds_every_requested_query_type(fake_query_llm, fake_huggingface_embeddings):
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

    query = transform_question(
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


def test_compute_queries_counts_multi_and_hyde_i_dont_know(fake_query_llm, tmp_path, capsys):
    """
    Input: two questions, the first with a bad multi reply and a HyDE refusal, and no saved queries.
    Output: the queries and what was printed.
    What it does: checks each question keeps its own phrasings and the I don't know counts are printed.
    """
    fake_query_llm.replies["Q one?"] = {"multi": "bad", "hyde": "I don't know."}
    fake_query_llm.replies["Q two?"] = {"multi": "aa\nbbbb\ncccccc", "hyde": "A passage."}
    query_args = make_query_args(query_types=[MULTI_QUERY, HYDE_QUERY])

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries(
        [make_eval_sample(0, "Q one?"), make_eval_sample(1, "Q two?")], make_eval_dataset_args(), str(tmp_path)
    )

    assert queries[0].multi_qus == []
    lengths = []
    for multi_qus in queries[1].multi_qus:
        lengths.append(multi_qus.embeddings[0])
    assert lengths == [2.0, 4.0, 6.0]
    printed = capsys.readouterr().out
    assert "Multi-query I don't know : 1 of 2" in printed
    assert "HyDE I don't know : 1 of 2" in printed


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

    val_queries, test_queries = QueryTransformation(query_args, make_ollama_args()).split_queries(queries)
    val_queries_again, test_queries_again = QueryTransformation(query_args, make_ollama_args()).split_queries(queries)

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

    val_queries, test_queries = QueryTransformation(query_args, make_ollama_args()).split_queries(queries)

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
        QueryTransformation(query_args, make_ollama_args()).split_queries(queries)


# ---------- the saved queries ----------


def test_saved_queries_load():
    """
    Input: the saved 200 Ollama queries copied into temp/.
    Output: the loaded queries.
    What it does: checks they load with all four query types and 3 phrasings each. Skipped where the folder is not
    present.
    """
    if not os.path.exists(SAVED_QUERY_PATH):
        pytest.skip(f"{SAVED_QUERY_PATH} is not present")

    queries = load_queries(SAVED_QUERY_PATH)

    multi_count = 0
    for query in queries:
        multi_count = multi_count + len(query.multi_qus)
    assert len(queries) == 200
    assert multi_count == 600
    assert queries[0].original_qus.embeddings.shape == (384,)
    assert queries[0].rewritten_qus is not None
    assert queries[0].hyde_ans is not None
