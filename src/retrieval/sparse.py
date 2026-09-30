import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunking.models import Chunk


class SparseRetriever:
    def __init__(
        self,
        chunks_dir: Path = Path(
            "data/processed/chunks"
        ),
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k

        self.chunks = self._load_chunks(
            chunks_dir
        )

        tokenized_corpus = [
            self._tokenize(
                self._searchable_text(chunk)
            )
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    def _load_chunks(
        self,
        chunks_dir: Path,
    ) -> list[Chunk]:

        chunks: list[Chunk] = []

        files = sorted(
            chunks_dir.glob("*_chunks.json")
        )

        for path in files:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            chunks.extend(
                Chunk.model_validate(item)
                for item in data
            )

        return chunks

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    @staticmethod
    def _searchable_text(
        chunk: Chunk,
    ) -> str:

        parts = [
            chunk.company,
            str(chunk.fiscal_year),
            chunk.major_section or "",
            " ".join(chunk.subsections),
            chunk.text,
        ]

        return " ".join(parts)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        limit = (
            top_k
            if top_k is not None
            else self.top_k
        )

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]

        return [
            (
                self.chunks[index],
                float(scores[index]),
            )
            for index in ranked_indices
        ]