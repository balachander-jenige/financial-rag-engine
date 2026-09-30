from src.generation.answer import AnswerGenerator
from src.generation.context import ContextBuilder
from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


def main():
    query = (
        "What was NVIDIA's fiscal 2026 revenue?"
    )

    embedding_service = EmbeddingService()
    store = QdrantStore()

    dense = DenseRetriever(
        embedding_service=embedding_service,
        store=store,
        top_k=20,
    )

    sparse = SparseRetriever(
        top_k=20,
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        top_k=10,
    )

    # -----------------------------
    # Retrieval
    # -----------------------------

    results = hybrid.retrieve(query)

    # -----------------------------
    # Context
    # -----------------------------

    context_builder = ContextBuilder(
        max_chunks=5,
    )

    context = context_builder.build(
        results
    )

    # -----------------------------
    # Generation
    # -----------------------------

    generator = AnswerGenerator()

    answer = generator.generate(
        query=query,
        context=context,
    )

    print("\nQUESTION:")
    print(query)

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()