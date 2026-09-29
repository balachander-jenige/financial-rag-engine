from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore
from src.retrieval.dense import DenseRetriever


def main():
    embedding_service = EmbeddingService()
    store = QdrantStore()

    retriever = DenseRetriever(
        embedding_service=embedding_service,
        store=store,
        top_k=5,
    )

    query = (
        "What was NVIDIA's fiscal 2026 revenue?"
    )

    results = retriever.retrieve(query)

    print(f"\nQUERY: {query}")
    print(f"RESULTS: {len(results)}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        payload = result.payload or {}

        print("\n" + "=" * 70)

        print("RANK:", rank)
        print("SCORE:", result.score)

        print(
            "COMPANY:",
            payload.get("company"),
        )

        print(
            "YEAR:",
            payload.get("fiscal_year"),
        )

        print(
            "MAJOR SECTION:",
            payload.get(
                "major_section"
            ),
        )

        print(
            "SUBSECTIONS:",
            payload.get("subsections"),
        )

        print(
            "PAGES:",
            payload.get("page_start"),
            "-",
            payload.get("page_end"),
        )

        print("\nTEXT PREVIEW:")

        print(
            payload.get(
                "text",
                "",
            )[:800]
        )


if __name__ == "__main__":
    main()