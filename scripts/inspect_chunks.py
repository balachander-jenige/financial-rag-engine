import json
from pathlib import Path

from src.chunking.structural import StructuralChunker
from src.ingestion.models import ParsedDocument


def main():
    path = Path(
        "data/processed/nvidia_2026_10k.json"
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    document = ParsedDocument.model_validate(data)

    chunker = StructuralChunker(
        max_chars=4000,
    )

    chunks = chunker.chunk(document)

    # ==========================================
    # Overall statistics
    # ==========================================

    sizes = [len(chunk.text) for chunk in chunks]

    print("\n===== CHUNK STATISTICS =====")

    print(f"Total chunks: {len(chunks)}")
    print(f"Minimum size: {min(sizes)}")
    print(f"Maximum size: {max(sizes)}")

    print(
        f"Average size: "
        f"{sum(sizes) / len(sizes):.0f}"
    )

    tiny = [
        size
        for size in sizes
        if size < 300
    ]

    small = [
        size
        for size in sizes
        if 300 <= size < 800
    ]

    medium = [
        size
        for size in sizes
        if 800 <= size < 2000
    ]

    large = [
        size
        for size in sizes
        if size >= 2000
    ]

    print(f"\nTiny (<300): {len(tiny)}")
    print(f"Small (300-799): {len(small)}")
    print(f"Medium (800-1999): {len(medium)}")
    print(f"Large (>=2000): {len(large)}")

    # ==========================================
    # Tiny chunks
    # ==========================================

    tiny_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.text) < 300
    ]

    print("\n===== TINY CHUNKS =====")

    for chunk in tiny_chunks[:20]:
        print(
            len(chunk.text),
            "| major:",
            chunk.major_section,
            "| subsections:",
            chunk.subsections,
            "| pages",
            chunk.page_start,
            "-",
            chunk.page_end,
        )

    # ==========================================
    # Inspect Item 7
    # ==========================================

    item7_chunks = [
        chunk
        for chunk in chunks
        if chunk.major_section
        and chunk.major_section.startswith(
            "Item 7."
        )
    ]

    print("\n===== ITEM 7 CHUNKS =====")
    print(f"Item 7 chunks: {len(item7_chunks)}")

    for chunk in item7_chunks:
        print("\n" + "=" * 70)

        print("CHUNK:", chunk.chunk_id)

        print(
            "MAJOR SECTION:",
            chunk.major_section,
        )

        print(
            "SUBSECTIONS:",
            chunk.subsections,
        )

        print(
            "PAGES:",
            chunk.page_start,
            "-",
            chunk.page_end,
        )

        print(
            "CHARS:",
            len(chunk.text),
        )

        print(
            "ELEMENTS:",
            len(chunk.element_ids),
        )

        print("\nTEXT PREVIEW:")
        print(chunk.text[:500])


if __name__ == "__main__":
    main()