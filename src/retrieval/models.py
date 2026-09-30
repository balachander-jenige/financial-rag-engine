from pydantic import BaseModel


class RetrievalResult(BaseModel):
    # Identity
    chunk_id: str
    document_id: str

    # Document metadata
    company: str
    fiscal_year: int

    # Structural metadata
    major_section: str | None = None
    subsections: list[str]

    # Provenance
    page_start: int | None = None
    page_end: int | None = None

    # Actual retrieved content
    text: str

    # Retrieval information
    score: float
    retrieval_method: str