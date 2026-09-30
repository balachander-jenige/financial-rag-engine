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
        print("\n" + "=" * 70)

        print("RANK:", rank)
        print("SCORE:", result.score)
        print("METHOD:", result.retrieval_method)

        print(
            "CHUNK ID:",
            result.chunk_id,
        )

        print(
            "COMPANY:",
            result.company,
        )

        print(
            "YEAR:",
            result.fiscal_year,
        )

        print(
            "MAJOR SECTION:",
            result.major_section,
        )

        print(
            "SUBSECTIONS:",
            result.subsections,
        )

        print(
            "PAGES:",
            result.page_start,
            "-",
            result.page_end,
        )

        print("\nTEXT PREVIEW:")
        print(result.text[:800])


if __name__ == "__main__":
    main()