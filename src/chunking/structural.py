import re

from src.ingestion.models import ParsedDocument

from .models import Chunk, StructuralSegment


ITEM_PATTERN = re.compile(
    r"^Item\s+\d+[A-Z]?\.",
    re.IGNORECASE,
)



def is_major_section(text: str | None) -> bool:
    if not text:
        return False

    return bool(ITEM_PATTERN.match(text.strip()))

def is_valid_heading(text: str | None) -> bool:
    if not text:
        return False

    text = text.strip()

    if len(text) < 2:
        return False

    alpha_count = sum(
        char.isalpha()
        for char in text
    )

    if alpha_count == 0:
        return False

    alpha_ratio = alpha_count / len(text)

    # Reject strings dominated by symbols / parser noise.
    if alpha_ratio < 0.5:
        return False

    return True
def is_valid_heading(text: str | None) -> bool:
    if not text:
        return False

    text = text.strip()

    if len(text) < 2:
        return False

    alpha_count = sum(
        char.isalpha()
        for char in text
    )

    if alpha_count == 0:
        return False

    alpha_ratio = alpha_count / len(text)

    # Reject strings dominated by symbols / parser noise.
    if alpha_ratio < 0.5:
        return False

    return True

class StructuralChunker:
    def __init__(
        self,
        max_chars: int = 4000,
    ) -> None:
        self.max_chars = max_chars

    # =========================================================
    # Stage 1: ParsedDocument -> StructuralSegments
    # =========================================================

    def _build_segments(
        self,
        document: ParsedDocument,
    ) -> list[StructuralSegment]:

        segments: list[StructuralSegment] = []

        current_major_section: str | None = None
        current_subsection: str | None = None

        for element in document.elements:

            # -----------------------------
            # Heading
            # -----------------------------

            if element.element_type == "heading":

                heading = element.text

                if not heading:
                    continue

                # Major SEC section:
                #
                # Item 1.
                # Item 1A.
                # Item 7.
                # Item 7A.
                # etc.

                if is_major_section(heading):
                    current_major_section = heading
                    current_subsection = None
                    continue

                if not is_valid_heading(heading):
                    continue

                current_subsection = heading

                continue

            # -----------------------------
            # Searchable content
            # -----------------------------

            if element.element_type not in {
                "text",
                "table",
            }:
                continue

            if not element.text:
                continue

            segment = StructuralSegment(
                major_section=current_major_section,
                subsection=current_subsection,
                text=element.text,
                page_number=element.page_number,
                element_id=element.element_id,
            )

            segments.append(segment)

        return segments

    # =========================================================
    # Stage 2: StructuralSegments -> Chunks
    # =========================================================

    def chunk(
        self,
        document: ParsedDocument,
    ) -> list[Chunk]:

        segments = self._build_segments(document)

        chunks: list[Chunk] = []

        current_segments: list[StructuralSegment] = []
        current_length = 0

        chunk_index = 0

        for segment in segments:

            # -------------------------------------------------
            # Hard boundary:
            # Don't cross major SEC Item boundaries.
            # -------------------------------------------------

            if (
                current_segments
                and segment.major_section
                != current_segments[0].major_section
            ):
                chunk = self._build_chunk(
                    document=document,
                    segments=current_segments,
                    chunk_index=chunk_index,
                )

                chunks.append(chunk)
                chunk_index += 1

                current_segments = []
                current_length = 0

            segment_length = len(segment.text)

            # -------------------------------------------------
            # Size boundary
            # -------------------------------------------------

            if (
                current_segments
                and current_length + segment_length
                > self.max_chars
            ):
                chunk = self._build_chunk(
                    document=document,
                    segments=current_segments,
                    chunk_index=chunk_index,
                )

                chunks.append(chunk)
                chunk_index += 1

                current_segments = []
                current_length = 0

            current_segments.append(segment)
            current_length += segment_length

        # -----------------------------------------------------
        # Flush final chunk
        # -----------------------------------------------------

        if current_segments:
            chunk = self._build_chunk(
                document=document,
                segments=current_segments,
                chunk_index=chunk_index,
            )

            chunks.append(chunk)

        return chunks

    # =========================================================
    # Build final Chunk
    # =========================================================

    def _build_chunk(
        self,
        document: ParsedDocument,
        segments: list[StructuralSegment],
        chunk_index: int,
    ) -> Chunk:

        text = "\n\n".join(
            segment.text
            for segment in segments
        )

        pages = [
            segment.page_number
            for segment in segments
            if segment.page_number is not None
        ]

        # Preserve subsection order while removing duplicates.
        subsections: list[str] = []

        for segment in segments:
            subsection = segment.subsection

            if (
                subsection
                and subsection not in subsections
            ):
                subsections.append(subsection)

        return Chunk(
            chunk_id=(
                f"{document.document.document_id}"
                f"_chunk_{chunk_index:06d}"
            ),

            document_id=document.document.document_id,

            company=document.document.company,
            fiscal_year=document.document.fiscal_year,
            filing_type=document.document.filing_type,
            source_file=document.document.source_file,

            major_section=segments[0].major_section,

            subsections=subsections,

            text=text,

            page_start=(
                min(pages)
                if pages
                else None
            ),

            page_end=(
                max(pages)
                if pages
                else None
            ),

            element_ids=[
                segment.element_id
                for segment in segments
            ],
        )