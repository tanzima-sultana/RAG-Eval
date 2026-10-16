"""
conftest.py

Shared fixtures for the qna tests.

FakeQnALLM stands in for the LLMs get_llm returns, and the RAGAS pieces
(wrappers, embeddings, transforms and the test set generator) are replaced
by fakes, so no LLM, API key, embedding model or GPU is needed and no
request is ever sent.
"""

from types import SimpleNamespace

import pytest

from common.args import EvalQnAArgs
from common.args import make_llm_args
from l1_ingestion.schema import Document
from l1_ingestion.schema import DocumentType
from l1_ingestion.schema import Metadata
from llm.schema import LLMCost
from qna import qna as qna_module
from qna.schema import EvalQnA


class FakeQnALLM:
    """
    Stand-in for the LLM get_llm returns: json_chat_model and compute_cost().
    """

    def __init__(self, llm_args):
        """
        Input: llm_args, the LLM settings.
        Output: nothing.
        What it does: stores the settings and a placeholder json_chat_model.
        """
        self.llm_args = llm_args
        self.json_chat_model = "json_chat_model of " + llm_args.llm_model
        self.cost_computed = False

    def compute_cost(self):
        """
        Input: none.
        Output: an LLMCost with no calls and a cost of 0.
        What it does: records that the cost was computed.
        """
        self.cost_computed = True
        return LLMCost(
            llm=self.llm_args.llm,
            llm_model=self.llm_args.llm_model,
            number_of_calls=0,
            input_tokens=0,
            output_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0
        )


class FakeHuggingFaceEmbeddings:
    """
    Stand-in for langchain_huggingface.HuggingFaceEmbeddings, so no model is loaded.
    """

    def __init__(self, model_name, model_kwargs, encode_kwargs):
        """
        Input: model_name, model_kwargs and encode_kwargs, as the real class takes them.
        Output: nothing.
        What it does: stores model_name.
        """
        self.model_name = model_name


class FakeTestsetGenerator:
    """
    Stand-in for ragas.testset.TestsetGenerator: returns the samples the test set up.
    """

    def __init__(self, llm, embedding_model, knowledge_graph, state):
        """
        Input: llm, embedding_model and knowledge_graph, as the real class takes them; state, the fixture's state.
        Output: nothing.
        What it does: records the knowledge graph it was given.
        """
        self.state = state
        state["knowledge_graphs"].append(knowledge_graph)
        state["question_llms"].append(llm)

    def generate(self, testset_size):
        """
        Input: testset_size, the number of samples asked for.
        Output: an object with the samples the test set in state["samples"].
        What it does: records testset_size and returns the samples.
        """
        self.state["testset_sizes"].append(testset_size)
        return SimpleNamespace(samples=self.state["samples"])


def make_ragas_sample(question, answer, contexts, synthesizer_name):
    """
    Input: question, answer, contexts (a list of passages, or None) and synthesizer_name.
    Output: an object shaped like one RAGAS test set sample.
    What it does: builds the eval_sample and synthesizer_name RAGAS returns for one item.
    """
    eval_sample = SimpleNamespace(user_input=question, reference=answer, reference_contexts=contexts)
    return SimpleNamespace(eval_sample=eval_sample, synthesizer_name=synthesizer_name)


@pytest.fixture
def ragas_sample():
    """
    Input: none.
    Output: the make_ragas_sample function.
    What it does: lets a test build RAGAS samples.
    """
    return make_ragas_sample


@pytest.fixture
def fake_ragas(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: a state dict: "samples" (set by the test), and the LLMs, knowledge graphs, transform runs, test set
    sizes and validated LLM settings recorded during the run.
    What it does: makes qna use FakeQnALLM and the fake RAGAS pieces, so generate_eval_qna runs without any model.
    """
    state = {
        "samples": [],
        "llms": [],
        "validated": [],
        "transform_runs": [],
        "knowledge_graphs": [],
        "question_llms": [],
        "testset_sizes": [],
    }

    def get_fake_llm(llm_args):
        """
        Input: llm_args, the LLM settings.
        Output: a new FakeQnALLM.
        What it does: stands in for llm.llm.get_llm.
        """
        fake_llm = FakeQnALLM(llm_args)
        state["llms"].append(fake_llm)
        return fake_llm

    def validate_fake_llm(llm_args):
        """
        Input: llm_args, the LLM settings.
        Output: True.
        What it does: stands in for llm.validate.validate_llm and records the settings it was given.
        """
        state["validated"].append(llm_args)
        return True

    def wrap_llm(chat_model, bypass_temperature):
        """
        Input: chat_model and bypass_temperature, as LangchainLLMWrapper takes them.
        Output: a tuple marking the wrapped model.
        What it does: stands in for ragas.llms.LangchainLLMWrapper.
        """
        return ("ragas llm", chat_model)

    def wrap_embeddings(embeddings):
        """
        Input: embeddings, the LangChain embeddings.
        Output: the same embeddings.
        What it does: stands in for ragas.embeddings.LangchainEmbeddingsWrapper.
        """
        return embeddings

    def make_transforms(documents, llm, embedding_model):
        """
        Input: documents, llm and embedding_model, as default_transforms takes them.
        Output: an empty list of transforms.
        What it does: stands in for ragas.testset.transforms.default_transforms.
        """
        return []

    def run_transforms(knowledge_graph, transforms):
        """
        Input: knowledge_graph and transforms, as apply_transforms takes them.
        Output: nothing.
        What it does: records that the knowledge graph was built, without changing it.
        """
        state["transform_runs"].append(knowledge_graph)

    def make_testset_generator(llm, embedding_model, knowledge_graph):
        """
        Input: llm, embedding_model and knowledge_graph, as TestsetGenerator takes them.
        Output: a FakeTestsetGenerator.
        What it does: stands in for ragas.testset.TestsetGenerator.
        """
        return FakeTestsetGenerator(llm, embedding_model, knowledge_graph, state)

    monkeypatch.setattr(qna_module, "get_llm", get_fake_llm)
    monkeypatch.setattr(qna_module, "validate_llm", validate_fake_llm)
    monkeypatch.setattr(qna_module, "LangchainLLMWrapper", wrap_llm)
    monkeypatch.setattr(qna_module, "LangchainEmbeddingsWrapper", wrap_embeddings)
    monkeypatch.setattr(qna_module, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)
    monkeypatch.setattr(qna_module, "default_transforms", make_transforms)
    monkeypatch.setattr(qna_module, "apply_transforms", run_transforms)
    monkeypatch.setattr(qna_module, "TestsetGenerator", make_testset_generator)
    return state


@pytest.fixture
def make_eval_qna_args():
    """
    Input: none.
    Output: a function that builds an EvalQnAArgs.
    What it does: returns a builder where any field can be overridden by keyword.
    """
    def build_eval_qna_args(**overrides):
        """
        Input: any EvalQnAArgs field as a keyword argument.
        Output: an EvalQnAArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {"eval_qna_model": "all-MiniLM-L6-v2", "n_sample_documents": 2, "n_sample_evals": 2}
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return EvalQnAArgs(**field_values)

    return build_eval_qna_args


@pytest.fixture
def make_qna_llm_args():
    """
    Input: none.
    Output: a function that builds an LLMArgs.
    What it does: returns a builder for one LLM's settings, by provider and model.
    """
    def build_llm_args(llm="claude", llm_model="claude-sonnet-5-5"):
        """
        Input: llm, the provider; llm_model, the model.
        Output: an LLMArgs with those and fixed prices and limits.
        What it does: builds the settings of one LLM.
        """
        return make_llm_args(llm, llm_model, 2.0, 10.0, 16000, 180)

    return build_llm_args


def make_long_text(word, number_of_words):
    """
    Input: word, the start of every word; number_of_words, how many words.
    Output: a text of numbered words, e.g. "alpha0 alpha1 ...".
    What it does: builds a text long enough, or short enough, for the RAGAS token limit.
    """
    words = []
    for word_number in range(number_of_words):
        words.append(f"{word}{word_number}")
    return " ".join(words)


@pytest.fixture
def long_text():
    """
    Input: none.
    Output: the make_long_text function.
    What it does: lets a test build texts over or under EVAL_MIN_DOCUMENT_TOKENS.
    """
    return make_long_text


@pytest.fixture
def make_document():
    """
    Input: none.
    Output: a function that builds a Document from an id, text and source.
    What it does: returns a builder that fills in minimal valid metadata.
    """
    def build_document(doc_id, text, source="file.pdf"):
        """
        Input: doc_id, text and source, the document's path.
        Output: a pdf Document with that id, text and source.
        What it does: builds the Document with placeholder metadata.
        """
        metadata = Metadata(source=source, file_size_bytes=1, content_hash="hash")
        return Document(id=doc_id, type=DocumentType.PDF, text=text, metadata=metadata)

    return build_document


@pytest.fixture
def make_eval_qna_item():
    """
    Input: none.
    Output: a function that builds an EvalQnA.
    What it does: returns a builder where any field can be overridden by keyword.
    """
    def build_eval_qna_item(**overrides):
        """
        Input: any EvalQnA field as a keyword argument.
        Output: an EvalQnA with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {
            "id": "eval_qna_0",
            "qus": "What is the refund policy?",
            "ans": "Refunds are given within 30 days.",
            "context": ["Refunds are given within 30 days of purchase."],
            "qus_type": "single_hop_specific_query_synthesizer",
            "doc_ids": ["doc_a"],
            "doc_paths": ["a.pdf"],
            "llm": "claude",
            "llm_model": "claude-sonnet-5-5",
        }
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return EvalQnA(**field_values)

    return build_eval_qna_item
