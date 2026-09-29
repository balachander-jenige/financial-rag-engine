import os
import time

import httpx
from dotenv import load_dotenv


load_dotenv()


class EmbeddingService:
    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set"
            )

        self.api_key = api_key

        self.model = (
            model
            or os.getenv(
                "EMBEDDING_MODEL",
                "openai/text-embedding-3-small",
            )
        )

        self.dimensions = (
            dimensions
            or int(
                os.getenv(
                    "EMBEDDING_DIMENSIONS",
                    "1536",
                )
            )
        )

        self.url = (
            "https://openrouter.ai/api/v1/embeddings"
        )

        self.client = httpx.Client(
            timeout=90.0,
        )

    def _request(
        self,
        texts: list[str],
        max_retries: int = 5,
    ) -> list[list[float]]:

        if not texts:
            return []

        for attempt in range(max_retries):

            try:
                response = self.client.post(
                    self.url,
                    headers={
                        "Authorization": (
                            f"Bearer {self.api_key}"
                        ),
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": texts,
                        "dimensions": self.dimensions,
                    },
                )

                response.raise_for_status()

                result = response.json()

                data = result.get("data")

                if not data:
                    raise RuntimeError(
                        "OpenRouter returned no embedding data"
                    )

                # Ensure vectors remain in the same order
                # as the input texts.
                data = sorted(
                    data,
                    key=lambda item: item["index"],
                )

                vectors = [
                    item["embedding"]
                    for item in data
                ]

                # Validate number of vectors.
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        "Embedding count does not match "
                        "input text count"
                    )

                # Validate vector dimensions.
                for vector in vectors:
                    if len(vector) != self.dimensions:
                        raise RuntimeError(
                            f"Expected "
                            f"{self.dimensions} dimensions, "
                            f"got {len(vector)}"
                        )

                return vectors

            except (
                httpx.HTTPError,
                RuntimeError,
            ) as exc:

                if attempt == max_retries - 1:
                    raise

                wait_seconds = 2 ** attempt

                print(
                    f"Embedding request failed: {exc}"
                )

                print(
                    f"Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

        raise RuntimeError(
            "Embedding request failed"
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return self._request(texts)

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        vectors = self._request(
            [query]
        )

        return vectors[0]