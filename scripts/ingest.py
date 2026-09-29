from pathlib import Path

from src.ingestion.loader import load_documents
from src.ingestion.parser import PDFParser


def main():
    documents = load_documents()

    print(f"Found {len(documents)} documents")

    output_dir = Path("data/processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create parser once and reuse it
    parser = PDFParser()

    # Process every discovered PDF
    for document in documents:
        print(f"\nParsing: {document.source_file}")

        try:
            parsed = parser.parse(document)

            output_path = (
                output_dir
                / f"{document.document_id}.json"
            )

            output_path.write_text(
                parsed.model_dump_json(indent=2),
                encoding="utf-8",
            )

            print(
                f"Parsed {len(parsed.elements)} elements"
            )

            print(
                f"Saved normalized document to: "
                f"{output_path}"
            )

        except Exception as exc:
            print(
                f"[ERROR] Failed to parse "
                f"{document.source_file}: {exc}"
            )


if __name__ == "__main__":
    main()