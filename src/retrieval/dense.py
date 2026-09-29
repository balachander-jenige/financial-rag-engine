from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore
from src.observability.tracing import tracer


class DenseRetriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        store: QdrantStore,
        top_k: int = 5,
    ) -> None:
        self.embedding_service = embedding_service
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
    ):
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        # ---------------------------------------
        # Main dense retrieval trace
        # ---------------------------------------

        with tracer.start_as_current_span(
            "dense_retrieval"
        ) as span:

            # Store useful information
            # about this retrieval request.
            span.set_attribute(
                "rag.query",
                query,
            )

            span.set_attribute(
                "rag.top_k",
                self.top_k,
            )

            # -----------------------------------
            # Query embedding
            # -----------------------------------

            with tracer.start_as_current_span(
                "query_embedding"
            ) as embedding_span:

                query_vector = (
                    self.embedding_service.embed_query(
                        query
                    )
                )

                embedding_span.set_attribute(
                    "embedding.dimensions",
                    len(query_vector),
                )

            # -----------------------------------
            # Qdrant vector search
            # -----------------------------------

            with tracer.start_as_current_span(
                "qdrant_search"
            ) as search_span:

                results = self.store.search(
                    query_vector=query_vector,
                    limit=self.top_k,
                )

                search_span.set_attribute(
                    "qdrant.result_count",
                    len(results),
                )

            # -----------------------------------
            # Retrieval result metadata
            # -----------------------------------

            span.set_attribute(
                "rag.result_count",
                len(results),
            )

            if results:
                span.set_attribute(
                    "rag.top_score",
                    float(results[0].score),
                )

                top_payload = (
                    results[0].payload or {}
                )

                company = top_payload.get(
                    "company"
                )

                if company:
                    span.set_attribute(
                        "rag.top_company",
                        company,
                    )

                chunk_id = top_payload.get(
                    "chunk_id"
                )

                if chunk_id:
                    span.set_attribute(
                        "rag.top_chunk_id",
                        chunk_id,
                    )

            return results