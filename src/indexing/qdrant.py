import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.chunking.models import Chunk


load_dotenv()


class QdrantStore:
    def __init__(
        self,
        vector_size: int | None = None,
    ) -> None:

        self.url = os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        )

        self.collection_name = os.getenv(
            "QDRANT_COLLECTION",
            "financial_filings_v1",
        )

        # Use explicitly supplied size if provided.
        # Otherwise read the embedding dimension
        # from the environment.
        self.vector_size = (
            vector_size
            if vector_size is not None
            else int(
                os.getenv(
                    "EMBEDDING_DIMENSIONS",
                    "1536",
                )
            )
        )

        self.client = QdrantClient(
            url=self.url,
        )

    def create_collection(self) -> None:
        """
        Create the Qdrant collection if it
        does not already exist.
        """

        if self.client.collection_exists(
            self.collection_name
        ):
            print(
                f"Collection already exists: "
                f"{self.collection_name}"
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created collection: "
            f"{self.collection_name}"
        )

        print(
            f"Vector size: "
            f"{self.vector_size}"
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Insert or update chunks and their vectors.

        Qdrant point IDs must be unsigned integers
        or UUIDs.

        We generate a deterministic UUID from the
        human-readable chunk_id so re-indexing the
        same chunk updates it instead of creating
        a duplicate.
        """

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks must match "
                "number of vectors"
            )

        points: list[PointStruct] = []

        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):
            # Validate embedding dimensions before
            # sending anything to Qdrant.
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Expected vector size "
                    f"{self.vector_size}, "
                    f"got {len(vector)} "
                    f"for chunk {chunk.chunk_id}"
                )

            # Deterministic Qdrant point ID.
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk.chunk_id,
                )
            )

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "company": chunk.company,
                    "fiscal_year": chunk.fiscal_year,
                    "filing_type": chunk.filing_type,
                    "source_file": chunk.source_file,
                    "major_section": chunk.major_section,
                    "subsections": chunk.subsections,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text": chunk.text,
                    "element_ids": chunk.element_ids,
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ):
        """
        Perform dense semantic search using
        cosine similarity.
        """

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Expected query vector size "
                f"{self.vector_size}, "
                f"got {len(query_vector)}"
            )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return result.points

    def get_collection_info(self):
        """
        Return information about the current
        Qdrant collection.
        """

        return self.client.get_collection(
            self.collection_name
        )