from src.generation.citations import CitationFormatter
from src.retrieval.models import RetrievalResult


def main():
    sources = [
        RetrievalResult(
            chunk_id=(
                "nvidia_2026_10k_chunk_000011"
            ),
            document_id="nvidia_2026_10k",
            company="NVIDIA",
            fiscal_year=2026,
            major_section=None,
            subsections=[
                "BUSINESS OVERVIEW",
            ],
            page_start=23,
            page_end=25,
            text="Fiscal 2026 revenue...",
            score=0.03,
            retrieval_method="hybrid",
        ),
        RetrievalResult(
            chunk_id=(
                "nvidia_2026_10k_chunk_000126"
            ),
            document_id="nvidia_2026_10k",
            company="NVIDIA",
            fiscal_year=2026,
            major_section=(
                "Item 7. Management's Discussion "
                "and Analysis of Financial Condition "
                "and Results of Operations"
            ),
            subsections=[
                "Fiscal Year 2026 Summary",
            ],
            page_start=124,
            page_end=125,
            text="Revenue was $215,938 million...",
            score=0.02,
            retrieval_method="hybrid",
        ),
    ]

    answer = (
        "NVIDIA's fiscal 2026 revenue was "
        "$215.9 billion, up 65% year on year "
        "[SOURCE 1][SOURCE 2]."
    )

    formatter = CitationFormatter()

    result = formatter.format(
        answer=answer,
        sources=sources,
    )

    print(result)


if __name__ == "__main__":
    main()