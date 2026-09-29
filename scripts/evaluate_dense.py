from src.evaluation.dataset import (
    load_retrieval_dataset,
)
from src.evaluation.retrieval import (
    evaluate_example,
)
from src.indexing.embeddings import (
    EmbeddingService,
)
from src.indexing.qdrant import QdrantStore
from src.retrieval.dense import DenseRetriever


def main():
    dataset = load_retrieval_dataset()

    print(
        f"Loaded {len(dataset)} "
        f"evaluation questions"
    )

    embedding_service = EmbeddingService()
    store = QdrantStore()

    # Retrieve 10 because we want to measure
    # Hit@1, @3, @5 and @10.
    retriever = DenseRetriever(
        embedding_service=embedding_service,
        store=store,
        top_k=10,
    )

    results = []

    for index, example in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"\n[{index}/{len(dataset)}] "
            f"{example.query}"
        )

        result = evaluate_example(
            example=example,
            retriever=retriever,
        )

        results.append(result)

        print(
            "First relevant rank:",
            result.first_relevant_rank,
        )

    total = len(results)

    hit_1 = sum(
        result.hit_at_1
        for result in results
    ) / total

    hit_3 = sum(
        result.hit_at_3
        for result in results
    ) / total

    hit_5 = sum(
        result.hit_at_5
        for result in results
    ) / total

    hit_10 = sum(
        result.hit_at_10
        for result in results
    ) / total

    mrr = sum(
        result.reciprocal_rank
        for result in results
    ) / total

    print("\n" + "=" * 60)
    print("DENSE RETRIEVAL V1")
    print("=" * 60)

    print(f"Questions: {total}")

    print(f"Hit@1:  {hit_1:.3f}")
    print(f"Hit@3:  {hit_3:.3f}")
    print(f"Hit@5:  {hit_5:.3f}")
    print(f"Hit@10: {hit_10:.3f}")

    print(f"MRR:    {mrr:.3f}")


if __name__ == "__main__":
    main()