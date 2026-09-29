from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class FinancialDocument(BaseModel):
    document_id: str
    company: str
    fiscal_year: int
    filing_type: str
    source_path: Path
    source_file: str


class BoundingBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class ParsedElement(BaseModel):
    element_id: str

    element_type: Literal[
        "heading",
        "text",
        "table",
        "picture",
        "other",
    ]

    text: str | None = None

    page_number: int | None = None
    bbox: BoundingBox | None = None


class ParsedDocument(BaseModel):
    document: FinancialDocument
    elements: list[ParsedElement]


class ParsedDocument(BaseModel):
    document: FinancialDocument
    elements: list[ParsedElement]