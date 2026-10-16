"""
prompt_template.py

The prompts the generation LLM gets for each eval question. The user prompt is filled with str.format().
"""

from common import constants

# The system prompts, numbered so earlier ones stay for comparison. Each is the same for every prompt.
# I_DONT_KNOW is the same constant l5 uses, so l10 can find the answers where the LLM did not know

# Prompt 0, the first: one short answer, or I_DONT_KNOW exactly
SYSTEM_PROMPT_0 = """You answer questions using only the context you are given.
Do not use any knowledge outside the context.
If the context does not hold the answer, reply with exactly: """ + constants.I_DONT_KNOW + """
Answer in a few sentences, with no introduction and no mention of the context."""

# Prompt 1: a complete answer, each case when the context holds several, a partial answer that says what is missing,
# and I_DONT_KNOW exactly only when nothing in the context is relevant
SYSTEM_PROMPT_1 = (
	"You answer questions using only the context you are given.\n"
	"Do not use any knowledge outside the context.\n"
	"Give a complete answer: include every detail the context gives that the question asks for, such as names, numbers and dates.\n"
	"If the question has several parts, answer each part, combining information from different parts of the context when needed.\n"
	"If the context holds different answers for different cases, such as different years, versions or documents, give each one and say which case it belongs to.\n"
	"If the context answers only part of the question, answer that part and say briefly which part the context does not cover. "
	"Do not start such an answer with \"" + constants.I_DONT_KNOW + "\".\n"
	"Only if the context holds nothing relevant to the question, reply with exactly: " + constants.I_DONT_KNOW + "\n"
	"Answer in plain sentences, with no introduction and no mention of the context."
)

# Prompt 2: prompt 1, but a reply saying the context holds nothing on the question must be I_DONT_KNOW exactly, so it is
# counted as I don't know instead of scored as a wrong answer
SYSTEM_PROMPT_2 = (
	"You answer questions using only the context you are given.\n"
	"Do not use any knowledge outside the context.\n"
	"Give a complete answer: include every detail the context gives that the question asks for, such as names, numbers and dates.\n"
	"If the question has several parts, answer each part, combining information from different parts of the context when needed.\n"
	"If the context holds different answers for different cases, such as different years, versions or documents, give each one and say which case it belongs to.\n"
	"If the context answers only part of the question, answer that part and say briefly which part the context does not cover. "
	"Do not start such an answer with \"" + constants.I_DONT_KNOW + "\".\n"
	"If the context holds nothing that answers any part of the question, reply with exactly: " + constants.I_DONT_KNOW + "\n"
	"Never reply only that the context or the text does not contain the answer; in that case reply with exactly: " + constants.I_DONT_KNOW + "\n"
	"Answer in plain sentences, with no introduction and no mention of the context."
)

# The system prompt l8 sends. Point it at another numbered prompt to switch
SYSTEM_PROMPT = SYSTEM_PROMPT_1

# Input: {context}, {question}. Output: the user prompt sent with SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}"""
