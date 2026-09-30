from pydantic import BaseModel


class RetrievalResult(BaseModel):
    chunk_id: str
    text: str

    company: str
    fiscal_year: int

    major_section: str | None = None
    subsections: list[str]

    page_start: int | None = None
    page_end: int | None = None

    score: float
    retrieval_method: str