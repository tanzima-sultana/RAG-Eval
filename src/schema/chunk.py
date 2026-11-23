import dataclasses

@dataclasses.dataclass
class Document:
	id: str
	text: str
	size: int
	type: str