import re

from src.retrieval.models import RetrievalResult


class CitationFormatter:
    SOURCE_PATTERN = re.compile(
        r"\[SOURCE\s+(\d+)\]",
        re.IGNORECASE,
    )

    def format(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> str:

        if not answer.strip():
            return answer

        source_numbers = self._extract_source_numbers(
            answer
        )

        if not source_numbers:
            return answer

        citations: list[str] = []

        for source_number in source_numbers:

            index = source_number - 1

            if index < 0 or index >= len(sources):
                continue

            source = sources[index]

            citation = self._format_source(
                source
            )

            if citation not in citations:
                citations.append(citation)

        if not citations:
            return answer

        citation_lines = "\n".join(
            f"- {citation}"
            for citation in citations
        )

        return (
            f"{answer}\n\n"
            f"Sources:\n"
            f"{citation_lines}"
        )

    def _extract_source_numbers(
        self,
        answer: str,
    ) -> list[int]:

        matches = self.SOURCE_PATTERN.findall(
            answer
        )

        source_numbers: list[int] = []

        for match in matches:
            number = int(match)

            if number not in source_numbers:
                source_numbers.append(number)

        return source_numbers

    def _format_source(
        self,
        source: RetrievalResult,
    ) -> str:

        pages = self._format_pages(
            source.page_start,
            source.page_end,
        )

        filing_type = self._filing_name(
            source.document_id
        )

        citation = (
            f"{source.company} "
            f"{source.fiscal_year} "
            f"{filing_type}"
        )

        if pages:
            citation += f", {pages}"

        if source.major_section:
            citation += (
                f", {source.major_section}"
            )

        return citation

    @staticmethod
    def _format_pages(
        page_start: int | None,
        page_end: int | None,
    ) -> str | None:

        if page_start is None:
            return None

        if (
            page_end is None
            or page_start == page_end
        ):
            return f"p. {page_start}"

        return (
            f"pp. {page_start}-{page_end}"
        )

    @staticmethod
    def _filing_name(
        document_id: str,
    ) -> str:

        if "10k" in document_id.lower():
            return "Form 10-K"

        return document_id