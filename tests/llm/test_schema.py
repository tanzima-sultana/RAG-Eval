"""
test_schema.py

Tests for llm.schema: LLMModel, LLMCost and LLMResponse.
"""

import dataclasses

from llm.schema import LLMCost
from llm.schema import LLMModel
from llm.schema import LLMResponse


def test_cost_and_response_are_llm_models():
    """
    Input: none.
    Output: LLMCost and LLMResponse are LLMModel subclasses.
    What it does: checks every LLM output names its provider and model.
    """
    assert issubclass(LLMCost, LLMModel)
    assert issubclass(LLMResponse, LLMModel)


def test_cost_is_one_flat_record():
    """
    Input: an LLMCost.
    Output: dataclasses.asdict gives one flat dict, with llm and llm_model next to the cost fields.
    What it does: checks the fields come by inheritance, not as a nested object, so logs and JSON stay flat.
    """
    llm_cost = LLMCost(llm="claude", llm_model="claude-haiku-5-5", number_of_calls=2, input_tokens=10, cache_creation_tokens=3,
                       cache_read_tokens=5, output_tokens=4, input_cost=0.1, cache_write_cost=0.01, cache_read_cost=0.02,
                       output_cost=0.2, total_cost=0.33)

    assert dataclasses.asdict(llm_cost) == {
        "llm": "claude", "llm_model": "claude-haiku-5-5", "number_of_calls": 2, "input_tokens": 10, "cache_creation_tokens": 3,
        "cache_read_tokens": 5, "output_tokens": 4, "input_cost": 0.1, "cache_write_cost": 0.01, "cache_read_cost": 0.02,
        "output_cost": 0.2, "total_cost": 0.33,
    }


def test_response_is_one_flat_record():
    """
    Input: an LLMResponse.
    Output: dataclasses.asdict gives one flat dict.
    What it does: checks the reply fields sit next to llm and llm_model.
    """
    llm_response = LLMResponse(llm="ollama", llm_model="llama3.1:8b", request_id="hash_a", text="An answer.",
                               input_tokens=7, output_tokens=3)

    assert dataclasses.asdict(llm_response) == {
        "llm": "ollama", "llm_model": "llama3.1:8b", "request_id": "hash_a", "text": "An answer.",
        "input_tokens": 7, "output_tokens": 3,
    }
