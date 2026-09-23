
import dataclasses
import datetime

@dataclasses.dataclass
class Document:
	id: str
	text: str
	title: str | None
	path: str
	format: str
	metadata: dict[str, object]
	content_hash: str
	timestamp: datetime.datetime
