import dataclasses


@dataclasses.dataclass
class Chunk:
	# {document_id}_{n}, with n running over the document, so the document id is the part before the first "_"
	chunk_id: str
	# The exact slice of the document's document_texts the chunk covers
	chunk_texts: str
	# Size in tokens of the embedding model, without the start and end special tokens
	chunk_size: int
	# fixed, sentence or semantic
	chunk_type: str
	# The original file path, from the file
	source: str
	# The pages the chunk covers, 2 or more when it runs across a page break
	page_numbers: list[int]
