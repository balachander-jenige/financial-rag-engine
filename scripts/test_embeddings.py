from src.indexing.embeddings import GeminiEmbeddingService


def main():
    service = GeminiEmbeddingService()

    texts = [
        "NVIDIA reported strong data center growth.",
        "Apple sells iPhone and Mac products.",
        "Microsoft provides Azure cloud services.",
    ]

    vectors = service.embed_documents(texts)

    print(f"Vectors returned: {len(vectors)}")

    for index, vector in enumerate(vectors):
        print(
            f"Vector {index}: "
            f"{len(vector)} dimensions"
        )

        print(
            "First 5 values:",
            vector[:5],
        )

    query_vector = service.embed_query(
        "What is NVIDIA's data center business?"
    )

    print(
        "\nQuery dimensions:",
        len(query_vector),
    )


if __name__ == "__main__":
    main()