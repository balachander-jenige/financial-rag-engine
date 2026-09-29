import json
from pathlib import Path

from src.chunking.structural import StructuralChunker
from src.ingestion.models import ParsedDocument


INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/chunks")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        path
        for path in INPUT_DIR.glob("*.json")
        if not path.name.endswith("_chunks.json")
    )

    print(f"Found {len(files)} parsed documents")

    chunker = StructuralChunker(
        max_chars=4000,
    )

    for path in files:
        print(f"\nChunking: {path.name}")

        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )

            document = ParsedDocument.model_validate(data)

            chunks = chunker.chunk(document)

            output_path = (
                OUTPUT_DIR
                / f"{document.document.document_id}_chunks.json"
            )

            output_data = [
                chunk.model_dump(mode="json")
                for chunk in chunks
            ]

            output_path.write_text(
                json.dumps(
                    output_data,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            print(
                f"Created {len(chunks)} chunks"
            )

            print(
                f"Saved: {output_path}"
            )

        except Exception as exc:
            print(
                f"[ERROR] Failed to chunk "
                f"{path.name}: {exc}"
            )


if __name__ == "__main__":
    main()