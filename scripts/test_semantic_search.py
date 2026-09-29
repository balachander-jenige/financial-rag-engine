import json
from pathlib import Path

from src.chunking.models import Chunk
from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore


def main():
    path = Path(
        "data/processed/chunks/"
        "nvidia_2026_10k_chunks.json"
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    all_chunks = [
        Chunk.model_validate(item)
        for item in data
    ]

    # Small test only.
    chunks = all_chunks[:5]

    print(f"Testing with {len(chunks)} chunks")

    for chunk in chunks:
      print(
          chunk.chunk_id,
          "characters:",
          len(chunk.text),
      )

    embedding_service = EmbeddingService()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    print("Generating embeddings...")

    vectors = embedding_service.embed_documents(
        texts
    )

    print(
        f"Generated {len(vectors)} vectors"
    )

    store = QdrantStore()
    store.create_collection()

    print("Uploading to Qdrant...")

    store.upsert_chunks(
        chunks=chunks,
        vectors=vectors,
    )

    print("Upload complete")

    # --------------------------------
    # Query
    # --------------------------------

    query = "What does Nvidia say about AI infrastructure?"

    print(f"\nQuery: {query}")

    query_vector = embedding_service.embed_query(
        query
    )

    results = store.search(
        query_vector=query_vector,
        limit=3,
    )

    print("\n===== RESULTS =====")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        payload = result.payload

        print("\n" + "=" * 70)

        print("RANK:", rank)
        print("SCORE:", result.score)

        print(
            "COMPANY:",
            payload.get("company"),
        )

        print(
            "PAGES:",
            payload.get("page_start"),
            "-",
            payload.get("page_end"),
        )

        print(
            "MAJOR SECTION:",
            payload.get("major_section"),
        )

        print(
            "SUBSECTIONS:",
            payload.get("subsections"),
        )

        print("\nTEXT:")
        print(
            payload.get("text", "")[:500]
        )


if __name__ == "__main__":
    main()