from src.evaluation.dataset import (
    load_retrieval_dataset,
)
from src.retrieval.sparse import SparseRetriever


def main():
    dataset = load_retrieval_dataset()

    print(
        f"Loaded {len(dataset)} "
        f"evaluation questions"
    )

    retriever = SparseRetriever(
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

        for rank, (chunk, score) in enumerate(
            results,
            start=1,
        ):
            if expected in chunk.text.lower():
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
    print("SPARSE RETRIEVAL V1")
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