import hashlib

from stages.question.query.schema import Query
from l7_context.schema import Context
from l8_prompt.schema import Prompt
from l8_prompt.prompt_template import SYSTEM_PROMPT
from l8_prompt.prompt_template import USER_PROMPT_TEMPLATE

def get_prompt_list(context_list: list[Context], queries: list[Query]) -> list[Prompt]:
	"""
	Input: context_list, the l7 contexts; queries, the queries of this run, tuning or held-out.
	Output: one Prompt per Context, in the same order, so prompt_list[i] belongs to context_list[i].
	What it does: fills the user prompt template with each context and its original question.
	An empty context gets an empty user prompt.
	"""
	prompt_list = []

	# 1. Store each original question by its id, so each Context finds its question by context.id
	question_by_id = {}
	for query in queries:
		question_by_id[query.id] = query.original_qus.qus

	# The system prompt is the same for every prompt, so count its words once
	system_prompt_word_count = len(SYSTEM_PROMPT.split())

	for context in context_list:
		# 2. Get the original question of this Context. The original is used for every query type,
		# because the rewritten, multi and hyde forms were only for searching
		question = question_by_id[context.id]

		# An empty context cannot hold the answer, so its user prompt stays empty and l9 skips the LLM call
		if context.context == "":
			prompt_list.append(Prompt(id=context.id, qus=question, system_prompt=SYSTEM_PROMPT, user_prompt="", token_count=0, prompt_hash=""))
			continue

		# 3. Fill the user prompt template with the context and the question
		user_prompt = USER_PROMPT_TEMPLATE.format(context=context.context, question=question)

		# 4. Estimate the token count: the context token count plus the words of the system prompt and the question
		question_word_count = len(question.split())
		token_count = context.token_count + system_prompt_word_count + question_word_count

		# 5. Hash the system prompt and user prompt, so l9 can find the answer of the same prompt in its cache
		prompt_text = SYSTEM_PROMPT + "\n\n" + user_prompt
		prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

		# 6. Keep the Prompt in the same place as its Context
		prompt = Prompt(
			id=context.id,
			qus=question,
			system_prompt=SYSTEM_PROMPT,
			user_prompt=user_prompt,
			token_count=token_count,
			prompt_hash=prompt_hash,
		)
		prompt_list.append(prompt)

	return prompt_list
