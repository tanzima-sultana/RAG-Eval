"""
test_query_transformation.py

Tests for langchain_rag.query_transformation. No LLM is called: the query LLM is replaced with a fake that answers
by prompt, and HuggingFaceEmbeddings with the fake from conftest.
"""

import json
import os
import types

import numpy as np
import pytest

from common import paths_temp
from common.args_temp import OllamaArgs
from common.args_temp import QnaArgs
from common.args_temp import QueryArgs
from common.constants import I_DONT_KNOW
from langchain_rag import query_transformation as query_transformation_module
from langchain_rag.query_transformation import QueryTransformation
from langchain_rag.query_transformation import load_queries
from langchain_rag.query_transformation import save_queries
from schema.llm import LLMArgs
from schema.qna import Qna
from schema.query import Ans
from schema.query import Query
from schema.query import Qus

SAVED_QUERY_PATH = (
    "temp/langchain/pdfa-eng-wds/queries/"
    "100_200_anthropic-claude-sonnet-5-5_all-MiniLM-L6-v2_all-MiniLM-L6-v2_ollama_chat-llama3.1-8b_3"
)


class FakeQueryLLMProvider:
    """
    Stand-in for LLMProvider. It answers each prompt type from the replies dict, keyed by the question text:
    replies[qus] = {"rewrite": ..., "multi": ..., "hyde": ...}. Every built model is recorded in built_models.
    """

    replies = {}
    built_models = []

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
        What it does: finds the question in the prompt and returns the reply of that prompt type.
        """
        question = user_prompt.split("Question: ", 1)[1]
        if user_prompt.startswith("Rewrite"):
            reply_type = "rewrite"
        elif user_prompt.startswith("Write exactly"):
            reply_type = "multi"
        else:
            reply_type = "hyde"
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
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryArgs(**field_values)


def make_qna_args():
    """
    Input: none.
    Output: a QnaArgs with test values.
    What it does: builds the QnA settings the query path is named by.
    """
    return QnaArgs(
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


def make_qna_item(item_number, question):
    """
    Input: item_number; question, the question text.
    Output: a Qna with id "eval_qna_{item_number}" and that question.
    What it does: builds one QnA item with placeholder values.
    """
    return Qna(
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


def make_query(query_id, multi_count):
    """
    Input: query_id; multi_count, the number of multi-query phrasings.
    Output: a Query with small distinct vectors.
    What it does: builds one Query to save and load.
    """
    base = float(len(query_id))
    multi_qus = []
    for multi_index in range(multi_count):
        multi_qus.append(Qus(f"multi {multi_index}", ["multi", str(multi_index)], np.array([base, 2.0, float(multi_index)], dtype=np.float32)))
    return Query(
        id=query_id,
        original_qus=Qus("original", ["original"], np.array([base, 0.0, 0.0], dtype=np.float32)),
        rewritten_qus=Qus("rewritten", ["rewritten"], np.array([base, 1.0, 0.0], dtype=np.float32)),
        multi_qus=multi_qus,
        hyde_ans=Ans("hyde passage", np.array([base, 3.0, 0.0], dtype=np.float32)),
        normalized=True,
        embedding_model="fake-model",
        llm="ollama_chat",
        llm_model="llama3.1:8b",
    )


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"embedding_model": " "}, "embedding_model"),
    ({"n_multi_qus": 0}, "n_multi_qus"),
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


def test_compute_queries_returns_saved_queries_without_building_an_llm(fake_query_llm, tmp_path, capsys):
    """
    Input: saved queries at the query path.
    Output: the queries, the built models and what was printed.
    What it does: checks the saved queries are returned and no LLM is built.
    """
    query_args = make_query_args()
    qna_args = make_qna_args()
    output_path = str(tmp_path)
    save_queries(paths_temp.get_query_path(output_path, qna_args, query_args), [make_query("eval_qna_0", 3)])

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries([], qna_args, output_path)

    assert len(queries) == 1
    assert fake_query_llm.built_models == []
    assert "Queries : 1 (all from cache)" in capsys.readouterr().out


def test_compute_queries_generates_and_saves_when_none_are_saved(fake_query_llm, tmp_path):
    """
    Input: one question and no saved queries.
    Output: the queries and the query folder.
    What it does: checks queries are generated with the query LLM and saved, so the next run loads them.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "What does X mean?", "multi": "a\nb\nc", "hyde": "X is a thing."}
    query_args = make_query_args()
    qna_args = make_qna_args()
    output_path = str(tmp_path)

    queries = QueryTransformation(query_args, make_ollama_args()).compute_queries([make_qna_item(0, "What is X?")], qna_args, output_path)

    assert fake_query_llm.built_models == ["ollama_chat/llama3.1:8b"]
    assert load_queries(paths_temp.get_query_path(output_path, qna_args, query_args))[0].rewritten_qus.qus == queries[0].rewritten_qus.qus


# ---------- generate_queries ----------


def test_generate_queries_transforms_each_question(fake_query_llm):
    """
    Input: one question, with a rewrite, 3 phrasings with a blank line between them, and a HyDE passage.
    Output: the Query.
    What it does: checks the texts, BM25 tokens, model fields and that each text gets its own vector.
    """
    fake_query_llm.replies["What is X?"] = {
        "rewrite": " What does X mean? ",
        "multi": "Define X.\n\nExplain X.\nWhat is meant by X?",
        "hyde": "X is a thing used for Y.",
    }

    queries = QueryTransformation(make_query_args(), make_ollama_args()).generate_queries([make_qna_item(0, "What is X?")])

    query = queries[0]
    assert query.id == "eval_qna_0"
    assert query.original_qus.qus == "What is X?"
    assert query.original_qus.tokens == ["what", "is", "x?"]
    assert query.rewritten_qus.qus == "What does X mean?"
    multi_texts = []
    for multi_qus in query.multi_qus:
        multi_texts.append(multi_qus.qus)
    assert multi_texts == ["Define X.", "Explain X.", "What is meant by X?"]
    assert query.hyde_ans.ans == "X is a thing used for Y."
    # The fake embeds a text as [len(text), 1, 0, 0]
    assert query.multi_qus[1].embeddings[0] == len("Explain X.")
    assert query.hyde_ans.embeddings[0] == len("X is a thing used for Y.")
    assert query.llm == "ollama_chat"
    assert query.llm_model == "llama3.1:8b"
    assert query.normalized is True


def test_generate_queries_drops_phrasings_when_line_count_is_wrong(fake_query_llm, capsys):
    """
    Input: a multi-query reply with 2 lines instead of 3.
    Output: the Query and what was printed.
    What it does: checks no phrasings are kept and the I don't know is counted.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "r", "multi": "only one\nonly two", "hyde": "h"}

    queries = QueryTransformation(make_query_args(), make_ollama_args()).generate_queries([make_qna_item(0, "What is X?")])

    assert queries[0].multi_qus == []
    assert "Multi-query I don't know : 1 of 1" in capsys.readouterr().out


@pytest.mark.parametrize("hyde_reply", ["I don't know.", "I do not know the answer", "I don’t know"])
def test_generate_queries_stores_hyde_refusal_as_i_dont_know(fake_query_llm, hyde_reply):
    """
    Input: a HyDE reply that refuses.
    Output: the Query.
    What it does: checks the refusal is stored as exactly I_DONT_KNOW, so retrieval can skip it.
    """
    fake_query_llm.replies["What is X?"] = {"rewrite": "r", "multi": "a\nb\nc", "hyde": hyde_reply}

    queries = QueryTransformation(make_query_args(), make_ollama_args()).generate_queries([make_qna_item(0, "What is X?")])

    assert queries[0].hyde_ans.ans == I_DONT_KNOW


def test_generate_queries_keeps_multi_vectors_with_their_question(fake_query_llm):
    """
    Input: two questions, the first with no phrasings kept and the second with 3.
    Output: the Queries.
    What it does: checks the flat list of phrasing vectors is split back to the right question.
    """
    fake_query_llm.replies["Q one?"] = {"rewrite": "r1", "multi": "bad", "hyde": "h1"}
    fake_query_llm.replies["Q two?"] = {"rewrite": "r2", "multi": "aa\nbbbb\ncccccc", "hyde": "h2"}

    queries = QueryTransformation(make_query_args(), make_ollama_args()).generate_queries(
        [make_qna_item(0, "Q one?"), make_qna_item(1, "Q two?")]
    )

    assert queries[0].multi_qus == []
    lengths = []
    for multi_qus in queries[1].multi_qus:
        lengths.append(multi_qus.embeddings[0])
    assert lengths == [2.0, 4.0, 6.0]


# ---------- the saved queries ----------


def test_saved_queries_load():
    """
    Input: the saved 200 Ollama queries copied into temp/.
    Output: the loaded queries.
    What it does: checks they load with 3 phrasings each. Skipped where the folder is not present.
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
