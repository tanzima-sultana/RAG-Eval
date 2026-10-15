from l5_query.schema import Query
from l7_context.schema import Context
from l8_prompt.schema import Prompt
from l8_prompt.prompt_template import SYSTEM_PROMPT
from l8_prompt.prompt_template import USER_PROMPT_TEMPLATE

def get_prompt_list(context_list: list[Context], tuning_queries: list[Query]) -> list[Prompt]:
	"""
	Input: context_list, the l7 contexts; tuning_queries, the l5 tuning queries.
	Output: one Prompt per Context, in the same order, so prompt_list[i] belongs to context_list[i].
	What it does: fills the user prompt template with each context and its original question.
	"""
	prompt_list = []

	# 1. Store each original question by its id, so each Context finds its question by context.id
	question_by_id = {}
	for query in tuning_queries:
		question_by_id[query.id] = query.original_qus.qus

	# The system prompt is the same for every prompt, so count its words once
	system_prompt_word_count = len(SYSTEM_PROMPT.split())

	for context in context_list:
		# 2. Get the original question of this Context. The original is used for every query type,
		# because the rewritten, multi and hyde forms were only for searching
		question = question_by_id[context.id]

		# 3. Fill the user prompt template with the context and the question
		user_prompt = USER_PROMPT_TEMPLATE.format(context=context.context, question=question)

		# 4. Estimate the token count: the context token count plus the words of the system prompt and the question
		question_word_count = len(question.split())
		token_count = context.token_count + system_prompt_word_count + question_word_count

		# 5. Keep the Prompt in the same place as its Context
		prompt = Prompt(
			id=context.id,
			system_prompt=SYSTEM_PROMPT,
			user_prompt=user_prompt,
			token_count=token_count,
		)
		prompt_list.append(prompt)

	return prompt_list
