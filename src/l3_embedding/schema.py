import dataclasses
import numpy as np

@dataclasses.dataclass
class Embedded:
    chunk_id: str
    embeddings: np.ndarray
    normalized: bool