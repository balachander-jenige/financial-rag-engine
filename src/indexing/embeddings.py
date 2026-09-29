import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class GeminiEmbeddingService:
    def __init__(
        self,
        model: str = "gemini-embedding-2",
        dimensions: int = 768,
    ) -> None:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model
        self.dimensions = dimensions

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        vectors: list[list[float]] = []

        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimensions,
                ),
            )

            vectors.append(
                result.embeddings[0].values
            )

        return vectors

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        result = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )

        return result.embeddings[0].values