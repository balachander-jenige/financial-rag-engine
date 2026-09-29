from pathlib import Path

from .models import FinancialDocument


def create_document_from_path(path: Path) -> FinancialDocument:
    parts = path.stem.split("-")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid filename: {path.name}. "
            "Expected COMPANY-YEAR-FILING.pdf"
        )

    company, fiscal_year, filing_type = parts

    if not fiscal_year.isdigit():
        raise ValueError(
            f"Invalid fiscal year in filename: {path.name}"
        )

    document_id = (
        f"{company}_{fiscal_year}_{filing_type}"
        .lower()
    )

    return FinancialDocument(
        document_id=document_id,
        company=company.upper(),
        fiscal_year=int(fiscal_year),
        filing_type=filing_type.upper(),
        source_path=path,
        source_file=path.name,
    )