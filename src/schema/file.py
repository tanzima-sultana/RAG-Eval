from enum import Enum

from pydantic import BaseModel, Field


class FileType(str, Enum):
	"""
	The supported document types. There is one value per parser in parsers/.
	Each value is a plain string, so it is saved in the output JSON as text,
	for example "pdf".
	"""
	PDF = "pdf"
	SCANNED_PDF = "scanned_pdf"
	DOCX = "docx"
	PPTX = "pptx"
	XLSX = "xlsx"
	CSV = "csv"
	HTML = "html"
	MD = "md"
	TXT = "txt"


class Metadata(BaseModel):
	file_size_bytes: int | None = Field(..., description="Size of the raw input file in bytes")
	content_hash: str | None = Field(..., description="SHA-256 hash of the raw file bytes, for exact deduplication")
	minhash_signature: list[int] | None = Field(..., description="MinHash/LSH signature for near-duplicate detection, None until it is computed")
	page_count: int | None = Field(..., description="Number of pages in the file, blank pages included (None for types without pages)")
	parser: str | None = Field(..., description="Library that extracted the text (e.g., pypdf, pypdfium2)")
	author: str | None = Field(None, description="Author from the file's own metadata, None if it has none")
	title: str | None = Field(None, description="Title from the file's own metadata, None if it has none")
	creation_date: str | None = Field(None, description="Creation date from the file's own metadata, None if it has none")

class Page(BaseModel):
	page_id: str = Field(..., description="Unique identifier of the page: {document_id}_{page_number}")
	page_number: int = Field(..., description="0-based page number within its file")
	page_texts: str = Field(..., description="Text content of the page")

class File(BaseModel):
	document_id: str = Field(..., description="Unique identifier of the file: the first 16 characters of the SHA-256 of its raw bytes")
	document_type: FileType = Field(..., description="Detected document type (e.g., pdf, docx)")
	document_texts: str = Field(..., description="Text content of the whole document")
	source: str = Field(..., description="Original input file path or URL")
	metadata: Metadata = Field(..., description="File-level metadata")
	pages: dict[int, Page] = Field(..., description="The pages of the file, keyed by page number")
