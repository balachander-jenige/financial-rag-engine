import json
from pathlib import Path

from src.chunking.models import Chunk
from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore


CHUNKS_DIR = Path("data/processed/chunks")

BATCH_SIZE = 10


def load_all_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []

    files = sorted(CHUNKS_DIR.glob("*_chunks.json"))

    print(f"Found {len(files)} chunk files")

    for path in files:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        document_chunks = [
            Chunk.model_validate(item)
            for item in data
        ]

        print(
            f"{path.name}: "
            f"{len(document_chunks)} chunks"
        )

        chunks.extend(document_chunks)

    return chunks


def main():
    chunks = load_all_chunks()

    print(f"\nTotal chunks: {len(chunks)}")

    embedding_service = EmbeddingService()

    store = QdrantStore()
    store.create_collection()

    total_batches = (
        len(chunks) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for start in range(
        0,
        len(chunks),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(chunks),
        )

        batch = chunks[start:end]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        print(
            f"\nBatch "
            f"{batch_number}/{total_batches}"
        )

        print(
            f"Chunks {start + 1}-{end}"
        )

        texts = [
            chunk.text
            for chunk in batch
        ]

        # -----------------------------
        # Generate embeddings
        # -----------------------------

        vectors = (
            embedding_service.embed_documents(
                texts
            )
        )

        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding count does not "
                "match chunk count"
            )

        # -----------------------------
        # Store in Qdrant
        # -----------------------------

        store.upsert_chunks(
            chunks=batch,
            vectors=vectors,
        )

        print(
            f"Indexed {end}/"
            f"{len(chunks)} chunks"
        )

    print("\nIndexing complete.")

    info = store.client.get_collection(
        store.collection_name
    )

    print(
        f"Qdrant points: "
        f"{info.points_count}"
    )


if __name__ == "__main__":
    main()