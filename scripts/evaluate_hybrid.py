from src.evaluation.dataset import load_retrieval_dataset
from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


def main():
    dataset = load_retrieval_dataset()

    print(
        f"Loaded {len(dataset)} "
        f"evaluation questions"
    )

    embedding_service = EmbeddingService()
    store = QdrantStore()

    # Get a larger candidate pool from each retriever.
    dense_retriever = DenseRetriever(
        embedding_service=embedding_service,
        store=store,
        top_k=20,
    )

    sparse_retriever = SparseRetriever(
        top_k=20,
    )

    # Return 10 final hybrid results because
    # we want to calculate Hit@10.
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k=10,
    )

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0

    reciprocal_rank_sum = 0.0

    for index, example in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"\n[{index}/{len(dataset)}] "
            f"{example.query}"
        )

        results = retriever.retrieve(
            example.query
        )

        expected = (
            example.expected_text
            .lower()
        )

        first_relevant_rank = None

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if expected in result.text.lower():
                first_relevant_rank = rank
                break

        print(
            "First relevant rank:",
            first_relevant_rank,
        )

        if first_relevant_rank is None:
            continue

        if first_relevant_rank <= 1:
            hit_1 += 1

        if first_relevant_rank <= 3:
            hit_3 += 1

        if first_relevant_rank <= 5:
            hit_5 += 1

        if first_relevant_rank <= 10:
            hit_10 += 1

        reciprocal_rank_sum += (
            1 / first_relevant_rank
        )

    total = len(dataset)

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL V1")
    print("=" * 60)

    print(f"Questions: {total}")

    print(
        f"Hit@1:  {hit_1 / total:.3f}"
    )

    print(
        f"Hit@3:  {hit_3 / total:.3f}"
    )

    print(
        f"Hit@5:  {hit_5 / total:.3f}"
    )

    print(
        f"Hit@10: {hit_10 / total:.3f}"
    )

    print(
        f"MRR:    "
        f"{reciprocal_rank_sum / total:.3f}"
    )


if __name__ == "__main__":
    main()