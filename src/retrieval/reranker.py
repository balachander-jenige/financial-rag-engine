from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 5,
    ):
        if not candidates:
            return []

        # Create query-document pairs
        pairs = [
            [query, candidate.text]
            for candidate in candidates
        ]

        # Cross-encoder relevance scores
        scores = self.model.predict(pairs)

        # Keep original candidate objects
        scored_candidates = list(
            zip(candidates, scores)
        )

        # Highest relevance score first
        scored_candidates.sort(
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # Return original result objects
        return [
            candidate
            for candidate, _ in scored_candidates[:top_k]
        ]