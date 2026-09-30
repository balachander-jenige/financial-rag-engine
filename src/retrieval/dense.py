from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant import QdrantStore
from src.observability.tracing import tracer
from src.retrieval.models import RetrievalResult


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
    ) -> list[RetrievalResult]:

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

                qdrant_results = self.store.search(
                    query_vector=query_vector,
                    limit=self.top_k,
                )

                search_span.set_attribute(
                    "qdrant.result_count",
                    len(qdrant_results),
                )

            # -----------------------------------
            # Convert Qdrant results into
            # our common RetrievalResult model
            # -----------------------------------

            results: list[RetrievalResult] = []

            for result in qdrant_results:

                payload = result.payload or {}

                retrieval_result = RetrievalResult(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    company=payload["company"],
                    fiscal_year=payload["fiscal_year"],

                    major_section=payload.get(
                        "major_section"
                    ),

                    subsections=payload.get(
                        "subsections",
                        [],
                    ),

                    page_start=payload.get(
                        "page_start"
                    ),

                    page_end=payload.get(
                        "page_end"
                    ),

                    text=payload.get(
                        "text",
                        "",
                    ),

                    score=float(
                        result.score
                    ),

                    retrieval_method="dense",
                )

                results.append(
                    retrieval_result
                )

            # -----------------------------------
            # Phoenix metadata
            # -----------------------------------

            span.set_attribute(
                "rag.result_count",
                len(results),
            )

            if results:

                span.set_attribute(
                    "rag.top_score",
                    results[0].score,
                )

                span.set_attribute(
                    "rag.top_company",
                    results[0].company,
                )

                span.set_attribute(
                    "rag.top_chunk_id",
                    results[0].chunk_id,
                )

            return results