from src.generation.context import ContextBuilder
from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever

# Import your actual reranker class here.


def main():
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

    query = (
        "What was NVIDIA's fiscal 2026 revenue?"
    )

    # For the first ContextBuilder test,
    # hybrid results are sufficient.
    # Once verified, plug in your reranker here.
    results = hybrid.retrieve(query)

    builder = ContextBuilder(
        max_chunks=5,
    )

    context = builder.build(
        results
    )

    print("\nQUESTION:")
    print(query)

    print("\nCONTEXT:")
    print(context)


if __name__ == "__main__":
    main()