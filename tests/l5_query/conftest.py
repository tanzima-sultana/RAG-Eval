"""
conftest.py

Shared fakes and fixtures for the l5_query test suite.

FakeQueryLLM stands in for the query LLM, so no Ollama server or paid API
is called. It answers each prompt from the replies it is given, and
records every prompt. FakeSentenceTransformer embeds a text from its
length and word count, so different test texts get different vectors.
"""

import numpy as np
import pytest

from common.args import EvalQnAArgs
from common.args import QueryArgs
from l5_query import query as query_module
from l5_query.schema import Ans
from l5_query.schema import Qus
from l5_query.schema import Query
from qna.schema import EvalQnA


def get_prompt_question(prompt):
    """
    Input: prompt, a filled l5 prompt.
    Output: the question the prompt was filled with.
    What it does: returns the text after "Question: ".
    """
    return prompt.split("Question: ")[1]


class FakeQueryLLM:
    """
    Stand-in for the LLM get_llm returns: generate() and compute_cost().
    """

    def __init__(self, multi_replies=None, hyde_replies=None):
        """
        Input: multi_replies and hyde_replies, optional dicts of question to reply.
        Output: nothing.
        What it does: stores the replies. A question with no reply gets a default one.
        """
        self.multi_replies = multi_replies or {}
        self.hyde_replies = hyde_replies or {}
        self.prompts = []
        self.cost_computed = False

    def generate(self, prompt):
        """
        Input: prompt, a filled rewrite, multi-query or HyDE prompt.
        Output: the reply for that prompt and question.
        What it does: records the prompt and picks the reply by the prompt's first words.
        """
        self.prompts.append(prompt)
        question = get_prompt_question(prompt)

        if prompt.startswith("Rewrite"):
            return f"  Rewritten {question}  "

        if prompt.startswith("Write exactly"):
            if question in self.multi_replies:
                return self.multi_replies[question]
            return f"First {question}\nSecond phrasing of {question}\nThird and longest phrasing of {question}"

        if question in self.hyde_replies:
            return self.hyde_replies[question]
        return f"A passage answering {question}"

    def compute_cost(self):
        """
        Input: none.
        Output: nothing.
        What it does: records that the cost was printed.
        """
        self.cost_computed = True


def fake_embed(text):
    """
    Input: text, one string.
    Output: a unit vector of length 3.
    What it does: embeds text from its length and word count.
    """
    vector = np.array([float(len(text)), float(len(text.split())), 1.0], dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FakeSentenceTransformer:
    """
    Stand-in for sentence_transformers.SentenceTransformer: only encode().
    """

    def __init__(self, model_name=None, device=None):
        """
        Input: model_name and device, as the real model takes them.
        Output: nothing.
        What it does: stores model_name.
        """
        self.model_name = model_name

    def encode(self, sentences, batch_size=32, normalize_embeddings=False):
        """
        Input: sentences, a list of strings; batch_size and normalize_embeddings (ignored).
        Output: a numpy array with one vector per sentence.
        What it does: embeds each sentence with fake_embed.
        """
        vectors = []
        for sentence in sentences:
            vectors.append(fake_embed(sentence))
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def fake_embed_function():
    """
    Input: none.
    Output: the fake_embed function.
    What it does: lets a test work out the vector FakeSentenceTransformer gives a text.
    """
    return fake_embed


@pytest.fixture
def fake_query_llm(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: a list that holds each FakeQueryLLM get_llm creates, and the
    replies to give it, set by the test before calling generate_queries.
    What it does: makes l5 use FakeQueryLLM and FakeSentenceTransformer.
    """
    created_llms = []
    replies = {"multi": {}, "hyde": {}}

    def get_fake_llm(llm_name, llm_model):
        """
        Input: llm_name and llm_model, as get_llm takes them.
        Output: a new FakeQueryLLM.
        What it does: stands in for llm.llm.get_llm.
        """
        fake_llm = FakeQueryLLM(replies["multi"], replies["hyde"])
        created_llms.append(fake_llm)
        return fake_llm

    monkeypatch.setattr(query_module, "get_llm", get_fake_llm)
    monkeypatch.setattr(query_module, "SentenceTransformer", FakeSentenceTransformer)
    return created_llms, replies


@pytest.fixture
def make_query_args():
    """
    Input: none.
    Output: a function that builds a QueryArgs.
    What it does: returns a builder where any field can be overridden by keyword.
    """
    def build_query_args(**overrides):
        """
        Input: any QueryArgs field as a keyword argument.
        Output: a QueryArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
        field_values["embedding_model"] = "fake-model"
        field_values["query_llm"] = "ollama"
        field_values["query_llm_model"] = "llama3.1:8b"
        field_values["n_multi_qus"] = 3
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return QueryArgs(**field_values)

    return build_query_args


@pytest.fixture
def make_eval_qna_args():
    """
    Input: none.
    Output: a function that builds an EvalQnAArgs.
    What it does: returns a builder with small defaults.
    """
    def build_eval_qna_args():
        """
        Input: none.
        Output: an EvalQnAArgs.
        What it does: builds the eval set settings the query folder is named by.
        """
        return EvalQnAArgs(eval_qna_llm="claude", eval_qna_model="claude-model", n_sample_documents=2, n_sample_evals=2)

    return build_eval_qna_args


@pytest.fixture
def make_eval_qna():
    """
    Input: none.
    Output: a function that builds EvalQnA items from questions.
    What it does: returns a builder giving each question the id "qna{i}".
    """
    def build_eval_qna(questions):
        """
        Input: questions, a list of question strings.
        Output: one EvalQnA per question.
        What it does: fills in placeholder answers and documents.
        """
        eval_qna = []
        for question_index in range(len(questions)):
            eval_qna.append(EvalQnA(
                id=f"qna{question_index}",
                qus=questions[question_index],
                ans="an answer",
                context=["a context"],
                qus_type="single_hop",
                doc_ids=["doc0"],
                doc_paths=["doc0.pdf"]
            ))
        return eval_qna

    return build_eval_qna


@pytest.fixture
def make_query():
    """
    Input: none.
    Output: a function that builds a Query.
    What it does: returns a builder with fake_embed vectors for every text.
    """
    def build_query(query_id, multi_qus_texts, hyde_text="A passage"):
        """
        Input: query_id; multi_qus_texts, the multi-query phrasings; hyde_text, the HyDE answer.
        Output: a Query.
        What it does: builds each Qus and Ans with lowercase tokens and fake_embed vectors.
        """
        original_text = f"Original {query_id}"
        rewritten_text = f"Rewritten question {query_id}"
        multi_qus = []
        for multi_qus_text in multi_qus_texts:
            multi_qus.append(Qus(multi_qus_text, multi_qus_text.lower().split(), fake_embed(multi_qus_text)))
        return Query(
            id=query_id,
            original_qus=Qus(original_text, original_text.lower().split(), fake_embed(original_text)),
            rewritten_qus=Qus(rewritten_text, rewritten_text.lower().split(), fake_embed(rewritten_text)),
            multi_qus=multi_qus,
            hyde_ans=Ans(hyde_text, fake_embed(hyde_text)),
            normalized=True,
            embedding_model="fake-model",
            llm_model="llama3.1:8b"
        )

    return build_query
