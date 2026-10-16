"""
test_qna.py

Tests for qna.qna.get_eval_qna and generate_eval_qna. The LLMs and RAGAS are
replaced by the fakes in conftest.py, so no model is loaded and no request is
sent.
"""

import os

import pytest

from common import paths
from qna import qna as qna_module
from qna.qna import generate_eval_qna
from qna.qna import get_eval_qna
from qna.qna import save_eval_qna


def make_long_documents(make_document, long_text):
    """
    Input: the make_document and long_text fixtures.
    Output: a dict of three documents (doc_a, doc_b, doc_c) over the RAGAS token limit, with sources a.pdf, b.pdf
    and c.pdf.
    What it does: builds documents every sample can come from.
    """
    documents = {}
    for doc_id, word, source in (("doc_a", "alpha", "a.pdf"), ("doc_b", "beta", "b.pdf"), ("doc_c", "gamma", "c.pdf")):
        documents[doc_id] = make_document(doc_id, long_text(word, 700), source)
    return documents


def passage_of(text, first_word, last_word):
    """
    Input: text, a document text; first_word and last_word, word positions.
    Output: those words of the text, joined with a newline and extra spaces.
    What it does: builds a passage the way RAGAS may return it, with its whitespace changed.
    """
    words = text.split()
    return "\n   ".join(words[first_word:last_word])


# ------ get_eval_qna

def test_cached_eval_qna_is_returned_without_any_llm(tmp_path, fake_ragas, monkeypatch, make_eval_qna_args, make_qna_llm_args, make_eval_qna_item):
    """
    Input: an eval QnA cache file for the settings and QnA model.
    Output: the cached items.
    What it does: checks a cached run validates no LLM and never generates, so it needs no API key.
    """
    eval_qna_args = make_eval_qna_args()
    qna_llm_args = make_qna_llm_args()
    save_eval_qna(str(tmp_path), eval_qna_args, qna_llm_args, [make_eval_qna_item()])

    def fail_generate(*arguments):
        """
        Input: anything.
        Output: never returns.
        What it does: fails the test if get_eval_qna tries to generate.
        """
        raise AssertionError("generate_eval_qna must not run when the cache exists")

    monkeypatch.setattr(qna_module, "generate_eval_qna", fail_generate)

    eval_qna = get_eval_qna(str(tmp_path), {}, eval_qna_args, qna_llm_args, make_qna_llm_args(llm_model="claude-haiku-5-5"))

    assert eval_qna == [make_eval_qna_item()]
    assert fake_ragas["validated"] == []
    assert fake_ragas["llms"] == []


def test_invalid_args_raise_before_the_cache_is_read(tmp_path, fake_ragas, make_eval_qna_args, make_qna_llm_args, make_eval_qna_item):
    """
    Input: n_sample_evals 0, with a cache file in the folder.
    Output: ValueError mentioning n_sample_evals.
    What it does: checks the settings are validated first, even when a cache could be loaded.
    """
    save_eval_qna(str(tmp_path), make_eval_qna_args(), make_qna_llm_args(), [make_eval_qna_item()])

    with pytest.raises(ValueError, match="n_sample_evals"):
        get_eval_qna(str(tmp_path), {}, make_eval_qna_args(n_sample_evals=0), make_qna_llm_args(), make_qna_llm_args())


def test_cache_miss_validates_both_llms_generates_and_saves(tmp_path, fake_ragas, monkeypatch, make_eval_qna_args, make_qna_llm_args, make_eval_qna_item):
    """
    Input: no cache file; generate_eval_qna replaced by one that returns one item.
    Output: that item, saved to the cache file, and both LLMs validated.
    What it does: checks a cache miss validates the QnA and knowledge graph LLMs, generates once, and saves, so the
    next call loads the cache.
    """
    eval_qna_args = make_eval_qna_args()
    qna_llm_args = make_qna_llm_args()
    qna_kg_llm_args = make_qna_llm_args(llm_model="claude-haiku-5-5")
    generate_calls = []

    def fake_generate(output_path, documents, args, question_llm_args, kg_llm_args):
        """
        Input: the arguments get_eval_qna passes to generate_eval_qna.
        Output: one EvalQnA.
        What it does: records the call and returns a fixed item.
        """
        generate_calls.append((question_llm_args, kg_llm_args))
        return [make_eval_qna_item()]

    monkeypatch.setattr(qna_module, "generate_eval_qna", fake_generate)

    eval_qna = get_eval_qna(str(tmp_path), {}, eval_qna_args, qna_llm_args, qna_kg_llm_args)

    assert eval_qna == [make_eval_qna_item()]
    assert fake_ragas["validated"] == [qna_llm_args, qna_kg_llm_args]
    assert generate_calls == [(qna_llm_args, qna_kg_llm_args)]
    assert os.path.exists(paths.get_eval_qna_path(str(tmp_path), eval_qna_args, "claude-sonnet-5-5"))

    # The second call loads the cache and does not generate again
    get_eval_qna(str(tmp_path), {}, eval_qna_args, qna_llm_args, qna_kg_llm_args)
    assert len(generate_calls) == 1


# ------ generate_eval_qna

def test_too_few_long_documents_raises_before_any_llm(tmp_path, fake_ragas, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: one long document and one short one, with n_sample_documents 2.
    Output: ValueError mentioning n_sample_documents.
    What it does: checks short documents are not counted, and the run stops before any LLM is built.
    """
    documents = {
        "doc_long": make_document("doc_long", long_text("alpha", 700)),
        "doc_short": make_document("doc_short", long_text("beta", 20)),
    }

    with pytest.raises(ValueError, match="n_sample_documents"):
        generate_eval_qna(str(tmp_path), documents, make_eval_qna_args(n_sample_documents=2), make_qna_llm_args(), make_qna_llm_args())

    assert fake_ragas["llms"] == []


def test_samples_become_eval_qna_linked_to_their_documents(tmp_path, fake_ragas, ragas_sample, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: three long documents; RAGAS returns a single-hop sample from doc_a with changed whitespace and a
    multi-hop sample from doc_b and doc_c with RAGAS's "<1-hop>" markers.
    Output: two EvalQnA with their question, answer, passages, type, ids eval_qna_0 and eval_qna_1, the
    documents and paths of their passages, and the QnA LLM.
    What it does: checks each RAGAS sample is converted, and each passage is found in its document despite the
    whitespace and hop markers.
    """
    documents = make_long_documents(make_document, long_text)
    passage_a = passage_of(documents["doc_a"].text, 100, 160)
    passage_b = passage_of(documents["doc_b"].text, 300, 360)
    passage_c = passage_of(documents["doc_c"].text, 0, 60)
    fake_ragas["samples"] = [
        ragas_sample("Question A?", "Answer A.", [passage_a], "single_hop_specific_query_synthesizer"),
        ragas_sample("Question BC?", "Answer BC.", ["<1-hop>\n\n" + passage_b, "<2-hop>\n\n" + passage_c], "multi_hop_abstract_query_synthesizer"),
    ]
    qna_llm_args = make_qna_llm_args()

    eval_qna = generate_eval_qna(str(tmp_path), documents, make_eval_qna_args(n_sample_documents=3), qna_llm_args, make_qna_llm_args(llm_model="claude-haiku-5-5"))

    assert len(eval_qna) == 2
    single_hop = eval_qna[0]
    assert single_hop.id == "eval_qna_0"
    assert single_hop.qus == "Question A?"
    assert single_hop.ans == "Answer A."
    assert single_hop.context == [passage_a]
    assert single_hop.qus_type == "single_hop_specific_query_synthesizer"
    assert single_hop.doc_ids == ["doc_a"]
    assert single_hop.doc_paths == ["a.pdf"]
    assert single_hop.llm == "claude"
    assert single_hop.llm_model == "claude-sonnet-5-5"

    multi_hop = eval_qna[1]
    assert multi_hop.id == "eval_qna_1"
    # The passages are kept as RAGAS wrote them, markers included
    assert multi_hop.context == ["<1-hop>\n\n" + passage_b, "<2-hop>\n\n" + passage_c]
    assert multi_hop.doc_ids == ["doc_b", "doc_c"]
    assert multi_hop.doc_paths == ["b.pdf", "c.pdf"]


def test_passage_found_in_no_document_gets_no_doc_id(tmp_path, fake_ragas, ragas_sample, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: a sample whose passage is not in any document, and a sample with no passages (None).
    Output: both items are kept, with empty doc_ids and doc_paths; the second has an empty context.
    What it does: checks an unmatched passage is not linked to a wrong document, and a sample without passages
    does not fail.
    """
    documents = make_long_documents(make_document, long_text)
    fake_ragas["samples"] = [
        ragas_sample("Question?", "Answer.", ["A passage that is in none of the documents at all."], "single_hop_specific_query_synthesizer"),
        ragas_sample("Question 2?", "Answer 2.", None, "single_hop_specific_query_synthesizer"),
    ]

    eval_qna = generate_eval_qna(str(tmp_path), documents, make_eval_qna_args(n_sample_documents=3), make_qna_llm_args(), make_qna_llm_args())

    assert eval_qna[0].doc_ids == []
    assert eval_qna[0].doc_paths == []
    assert eval_qna[1].context == []
    assert eval_qna[1].doc_ids == []


def test_same_passage_twice_links_its_document_once(tmp_path, fake_ragas, ragas_sample, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: a multi-hop sample whose two passages both come from doc_a.
    Output: doc_ids is ["doc_a"], once.
    What it does: checks a document is listed once per item, however many passages come from it.
    """
    documents = make_long_documents(make_document, long_text)
    passage_1 = passage_of(documents["doc_a"].text, 0, 50)
    passage_2 = passage_of(documents["doc_a"].text, 400, 450)
    fake_ragas["samples"] = [
        ragas_sample("Question?", "Answer.", ["<1-hop>\n\n" + passage_1, "<2-hop>\n\n" + passage_2], "multi_hop_specific_query_synthesizer"),
    ]

    eval_qna = generate_eval_qna(str(tmp_path), documents, make_eval_qna_args(n_sample_documents=3), make_qna_llm_args(), make_qna_llm_args())

    assert eval_qna[0].doc_ids == ["doc_a"]
    assert eval_qna[0].doc_paths == ["a.pdf"]


def test_each_llm_has_its_job_and_its_cost_is_computed(tmp_path, fake_ragas, ragas_sample, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: a QnA LLM (claude-sonnet-5-5) and a knowledge graph LLM (claude-haiku-5-5).
    Output: the test set generator gets the QnA LLM's json_chat_model, it is asked for n_sample_evals items, and
    the cost of both LLMs is computed.
    What it does: checks the two LLMs are not swapped and both are counted in the cost.
    """
    documents = make_long_documents(make_document, long_text)
    fake_ragas["samples"] = [ragas_sample("Question?", "Answer.", [], "single_hop_specific_query_synthesizer")]

    generate_eval_qna(str(tmp_path), documents, make_eval_qna_args(n_sample_documents=3, n_sample_evals=1), make_qna_llm_args(), make_qna_llm_args(llm_model="claude-haiku-5-5"))

    assert fake_ragas["question_llms"] == [("ragas llm", "json_chat_model of claude-sonnet-5-5")]
    assert fake_ragas["testset_sizes"] == [1]
    built_models = []
    for fake_llm in fake_ragas["llms"]:
        built_models.append(fake_llm.llm_args.llm_model)
        assert fake_llm.cost_computed
    assert sorted(built_models) == ["claude-haiku-5-5", "claude-sonnet-5-5"]


def test_knowledge_graph_is_built_once_and_then_loaded(tmp_path, fake_ragas, ragas_sample, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: two runs with the same settings and knowledge graph model.
    Output: the knowledge graph file exists after the first run, and the second run does not build it again.
    What it does: checks the knowledge graph is saved, so a failed or repeated run does not pay to build it again.
    """
    documents = make_long_documents(make_document, long_text)
    fake_ragas["samples"] = [ragas_sample("Question?", "Answer.", [], "single_hop_specific_query_synthesizer")]
    eval_qna_args = make_eval_qna_args(n_sample_documents=3)
    qna_kg_llm_args = make_qna_llm_args(llm_model="claude-haiku-5-5")

    generate_eval_qna(str(tmp_path), documents, eval_qna_args, make_qna_llm_args(), qna_kg_llm_args)
    kg_path = paths.get_eval_kg_path(str(tmp_path), eval_qna_args, "claude-haiku-5-5")
    assert os.path.exists(kg_path)
    assert not os.path.exists(kg_path + ".tmp")
    assert len(fake_ragas["transform_runs"]) == 1

    generate_eval_qna(str(tmp_path), documents, eval_qna_args, make_qna_llm_args(), qna_kg_llm_args)
    assert len(fake_ragas["transform_runs"]) == 1
    # The loaded graph holds the same three documents
    assert len(fake_ragas["knowledge_graphs"][1].nodes) == 3


def test_document_sample_is_reproducible_and_skips_short_documents(tmp_path, fake_ragas, ragas_sample, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: four long documents and one short one, sampled 2 at a time in two separate output folders.
    Output: both runs put the same two long documents in the knowledge graph.
    What it does: checks the fixed seed gives the same sample every run, and a short document is never sampled.
    """
    documents = {}
    for doc_id in ("doc_1", "doc_2", "doc_3", "doc_4"):
        documents[doc_id] = make_document(doc_id, long_text(doc_id, 700))
    documents["doc_short"] = make_document("doc_short", long_text("short", 20))
    fake_ragas["samples"] = [ragas_sample("Question?", "Answer.", [], "single_hop_specific_query_synthesizer")]

    sampled_doc_ids_per_run = []
    for run_folder in ("run_1", "run_2"):
        generate_eval_qna(str(tmp_path / run_folder), documents, make_eval_qna_args(n_sample_documents=2), make_qna_llm_args(), make_qna_llm_args())
        knowledge_graph = fake_ragas["knowledge_graphs"][-1]
        sampled_doc_ids = []
        for node in knowledge_graph.nodes:
            sampled_doc_ids.append(node.properties["document_metadata"]["doc_id"])
        sampled_doc_ids_per_run.append(sampled_doc_ids)

    assert sampled_doc_ids_per_run[0] == sampled_doc_ids_per_run[1]
    assert len(sampled_doc_ids_per_run[0]) == 2
    assert "doc_short" not in sampled_doc_ids_per_run[0]


def test_zero_samples_raises(tmp_path, fake_ragas, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: RAGAS returns no samples.
    Output: ValueError saying 0 eval QnA were generated.
    What it does: checks an empty result stops the run instead of saving an empty eval set.
    """
    documents = make_long_documents(make_document, long_text)
    fake_ragas["samples"] = []

    with pytest.raises(ValueError, match="generated 0 eval QnA"):
        generate_eval_qna(str(tmp_path), documents, make_eval_qna_args(n_sample_documents=3), make_qna_llm_args(), make_qna_llm_args())


def test_fewer_samples_than_asked_are_kept_with_a_warning(tmp_path, fake_ragas, ragas_sample, capsys, make_eval_qna_args, make_qna_llm_args, make_document, long_text):
    """
    Input: n_sample_evals 3, and RAGAS returns 1 sample.
    Output: one EvalQnA, and a printed warning.
    What it does: checks a short result is kept and reported, not dropped.
    """
    documents = make_long_documents(make_document, long_text)
    fake_ragas["samples"] = [ragas_sample("Question?", "Answer.", [], "single_hop_specific_query_synthesizer")]

    eval_qna = generate_eval_qna(str(tmp_path), documents, make_eval_qna_args(n_sample_documents=3, n_sample_evals=3), make_qna_llm_args(), make_qna_llm_args())

    assert len(eval_qna) == 1
    assert "Generated 1 of 3 eval QnA" in capsys.readouterr().out
