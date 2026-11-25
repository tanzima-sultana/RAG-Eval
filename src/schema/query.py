import dataclasses

@dataclasses.dataclass
class Query:
	query_type: str
	llm: str
	llm_model: str
	question_id: str
	original_question: str
	transformed_query: str
