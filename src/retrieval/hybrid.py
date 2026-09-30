from src.retrieval.dense import DenseRetriever
from src.retrieval.models import RetrievalResult
from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        # -----------------------------------
        # Retrieve candidates
        # -----------------------------------

        dense_results = (
            self.dense_retriever.retrieve(query)
        )

        sparse_results = (
            self.sparse_retriever.retrieve(query)
        )

        # -----------------------------------
        # RRF scores
        # -----------------------------------

        rrf_scores: dict[str, float] = {}

        # Keep one copy of each chunk.
        chunks: dict[str, RetrievalResult] = {}

        # Dense ranking
        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = result.chunk_id

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            chunks[chunk_id] = result

        # Sparse ranking
        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            chunk_id = result.chunk_id

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            # Keep the chunk if it wasn't already
            # returned by dense retrieval.
            if chunk_id not in chunks:
                chunks[chunk_id] = result

        # -----------------------------------
        # Sort by combined RRF score
        # -----------------------------------

        ranked_chunk_ids = sorted(
            rrf_scores,
            key=lambda chunk_id: rrf_scores[
                chunk_id
            ],
            reverse=True,
        )

        # -----------------------------------
        # Build final hybrid results
        # -----------------------------------

        results: list[RetrievalResult] = []

        for chunk_id in ranked_chunk_ids[
            : self.top_k
        ]:
            original = chunks[chunk_id]

            result = RetrievalResult(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                company=original.company,
                fiscal_year=original.fiscal_year,
                major_section=original.major_section,
                subsections=original.subsections,
                page_start=original.page_start,
                page_end=original.page_end,
                text=original.text,
                score=rrf_scores[chunk_id],
                retrieval_method="hybrid",
            )

            results.append(result)

        return results