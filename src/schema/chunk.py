import dataclasses

@dataclasses.dataclass
class Chunk:
	id: str
	text: str
	size: int
	type: str