"""
prompt.py

The prompts the query LLM gets to transform each eval question. Each prompt is filled with str.format().
The query cache does not record the prompts, so delete the outputs/<dataset>/queries folder after changing one.
"""

# Input: {qus}. Output: one rewritten question
REWRITE_PROMPT = """Rewrite the question below so it is clear and specific. Do not mention documents or searching.
Keep its meaning. Use the words a document answering it would likely use.
Return only the rewritten question, with no introduction, quotes or explanation.

Question: {qus}"""

# Input: {qus}, {n_multi_qus}. Output: n_multi_qus questions, one per line
MULTI_QUERY_PROMPT = """Write exactly {n_multi_qus} different phrasings of the question below.
You are only rephrasing the question, not answering it or searching for it, so never say you cannot find information.
Each phrasing must keep the meaning of the question but use different words.
Return only the {n_multi_qus} phrasings, one per line, with no numbering, bullets, introduction or explanation.

Question: {qus}"""

# Input: {qus}. Output: one hypothetical answer passage. It is only used to search, so it may hold invented facts
HYDE_PROMPT = """Write a short passage of 3 to 5 sentences that answers the question below, as it might appear in a document.
If you are not sure of the answer, still write a plausible passage, using the terms a real document on this topic would use.
Never say you do not know or lack information.
Return only the passage, with no introduction, title or explanation.

Question: {qus}"""
