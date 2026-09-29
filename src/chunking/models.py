from pydantic import BaseModel


class StructuralSegment(BaseModel):
    """
    A piece of document content with the structural context
    that was active when the content was encountered.
    """

    major_section: str | None = None
    subsection: str | None = None

    text: str

    page_number: int | None = None
    element_id: str


class Chunk(BaseModel):
    chunk_id: str
    document_id: str

    company: str
    fiscal_year: int
    filing_type: str
    source_file: str

    major_section: str | None = None

    # A chunk may contain content from multiple subsections.
    subsections: list[str]

    text: str

    page_start: int | None = None
    page_end: int | None = None

    element_ids: list[str]