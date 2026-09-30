from src.retrieval.models import RetrievalResult


class ContextBuilder:
    def __init__(
        self,
        max_chunks: int = 5,
    ) -> None:
        self.max_chunks = max_chunks

    def build(
        self,
        results: list[RetrievalResult],
    ) -> str:

        if not results:
            return ""

        selected_results = results[
            : self.max_chunks
        ]

        context_parts: list[str] = []

        for index, result in enumerate(
            selected_results,
            start=1,
        ):
            section = (
                result.major_section
                or "Unknown section"
            )

            subsections = (
                ", ".join(result.subsections)
                if result.subsections
                else "None"
            )

            pages = self._format_pages(
                result.page_start,
                result.page_end,
            )

            context = (
                f"[SOURCE {index}]\n"
                f"Company: {result.company}\n"
                f"Fiscal Year: "
                f"{result.fiscal_year}\n"
                f"Document ID: "
                f"{result.document_id}\n"
                f"Chunk ID: "
                f"{result.chunk_id}\n"
                f"Section: {section}\n"
                f"Subsections: {subsections}\n"
                f"Pages: {pages}\n"
                f"\n"
                f"{result.text}"
            )

            context_parts.append(context)

        return "\n\n".join(
            context_parts
        )

    @staticmethod
    def _format_pages(
        page_start: int | None,
        page_end: int | None,
    ) -> str:

        if page_start is None:
            return "Unknown"

        if (
            page_end is None
            or page_start == page_end
        ):
            return str(page_start)

        return (
            f"{page_start}-{page_end}"
        )