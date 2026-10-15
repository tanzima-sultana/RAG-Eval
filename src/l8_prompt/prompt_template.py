"""
prompt_template.py

The prompts the generation LLM gets for each eval question. The user prompt is filled with str.format().
"""

from common import constants

# The instructions for the LLM, the same for every prompt. Output: one answer, or I_DONT_KNOW exactly
# I_DONT_KNOW is the same constant l5 uses, so l10 can find the answers where the LLM did not know
SYSTEM_PROMPT = """You answer questions using only the context you are given.
Do not use any knowledge outside the context.
If the context does not hold the answer, reply with exactly: """ + constants.I_DONT_KNOW + """
Answer in a few sentences, with no introduction and no mention of the context."""

# Input: {context}, {question}. Output: the user prompt sent with SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}"""
