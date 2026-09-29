from docling.document_converter import DocumentConverter

from .models import (
    BoundingBox,
    FinancialDocument,
    ParsedDocument,
    ParsedElement,
)


class PDFParser:
    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def _table_to_markdown(self, item) -> str:
        """
        Convert a Docling table into Markdown text.

        Example:

        | Revenue | 2026 | 2025 |
        | --- | --- | --- |
        | Data Center | 100 | 80 |
        """

        data = getattr(item, "data", None)

        if data is None:
            return ""

        grid = getattr(data, "grid", None)

        if not grid:
            return ""

        rows: list[list[str]] = []

        for row in grid:
            values = []

            for cell in row:
                text = cell.text.strip() if cell.text else ""

                # Prevent Markdown table syntax from breaking
                text = text.replace("|", "\\|")

                values.append(text)

            rows.append(values)

        if not rows:
            return ""

        # Find largest number of columns
        column_count = max(len(row) for row in rows)

        # Make every row the same width
        normalized_rows = [
            row + [""] * (column_count - len(row))
            for row in rows
        ]

        lines = []

        # First row
        lines.append(
            "| " + " | ".join(normalized_rows[0]) + " |"
        )

        # Markdown separator
        lines.append(
            "| "
            + " | ".join(["---"] * column_count)
            + " |"
        )

        # Remaining rows
        for row in normalized_rows[1:]:
            lines.append(
                "| " + " | ".join(row) + " |"
            )

        return "\n".join(lines)

    def parse(
        self,
        document: FinancialDocument,
    ) -> ParsedDocument:

        result = self.converter.convert(document.source_path)

        elements: list[ParsedElement] = []

        for index, (item, level) in enumerate(
            result.document.iterate_items()
        ):
            label = str(getattr(item, "label", ""))

            # -------------------------
            # Determine element type
            # -------------------------

            if label == "section_header":
                element_type = "heading"

            elif label == "text":
                element_type = "text"

            elif label == "table":
                element_type = "table"

            elif label == "picture":
                element_type = "picture"

            else:
                element_type = "other"

            # -------------------------
            # Extract content
            # -------------------------

            if element_type == "table":
                text = self._table_to_markdown(item)
            else:
                text = getattr(item, "text", None)

            # -------------------------
            # Extract provenance
            # -------------------------

            page_number = None
            bbox = None

            provenance = getattr(item, "prov", None)

            if provenance:
                prov = provenance[0]

                page_number = prov.page_no

                if prov.bbox:
                    bbox = BoundingBox(
                        left=prov.bbox.l,
                        top=prov.bbox.t,
                        right=prov.bbox.r,
                        bottom=prov.bbox.b,
                    )

            # -------------------------
            # Build normalized element
            # -------------------------

            element = ParsedElement(
                element_id=(
                    f"{document.document_id}_{index:06d}"
                ),
                element_type=element_type,
                text=text,
                page_number=page_number,
                bbox=bbox,
            )

            elements.append(element)

        return ParsedDocument(
            document=document,
            elements=elements,
        )