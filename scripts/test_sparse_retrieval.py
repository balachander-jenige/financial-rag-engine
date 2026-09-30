from src.retrieval.sparse import SparseRetriever


def main():
    retriever = SparseRetriever(
        top_k=5,
    )

    query = (
        "What was NVIDIA's fiscal 2026 revenue?"
    )

    results = retriever.retrieve(query)

    print(f"\nQUERY: {query}")
    print(f"RESULTS: {len(results)}")

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 70)

        print("RANK:", rank)
        print("BM25 SCORE:", score)

        print(
            "COMPANY:",
            chunk.company,
        )

        print(
            "YEAR:",
            chunk.fiscal_year,
        )

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

        print("\nTEXT PREVIEW:")
        print(chunk.text[:800])


if __name__ == "__main__":
    main()