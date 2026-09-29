from pathlib import Path

from .metadata import create_document_from_path
from .models import FinancialDocument


def load_documents(
    raw_dir: Path = Path("data/raw"),
) -> list[FinancialDocument]:

    pdf_files = sorted(raw_dir.glob("*.pdf"))

    documents = []

    for pdf_path in pdf_files:
        document = create_document_from_path(pdf_path)
        documents.append(document)

    return documents