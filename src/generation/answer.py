import os

import httpx
from dotenv import load_dotenv


load_dotenv()


class AnswerGenerator:
    def __init__(
        self,
        model: str | None = None,
    ) -> None:

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set"
            )

        self.model = (
            model
            or os.getenv(
                "GENERATION_MODEL",
                "openrouter/free",
            )
        )

        self.url = (
            "https://openrouter.ai/api/v1/chat/completions"
        )

        self.client = httpx.Client(
            timeout=120.0,
        )

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if not context.strip():
            return (
                "I could not find sufficient "
                "evidence in the retrieved filings."
            )

        system_prompt = """
You are a financial filings question-answering assistant.

Answer the user's question using ONLY the provided sources.

Rules:
1. Do not use outside knowledge.
2. Do not invent financial figures.
3. If the sources do not contain enough information, say so.
4. Prefer exact reported figures over rounded figures.
5. Keep the answer concise and factual.
6. Cite supporting sources using [SOURCE N].
7. Every material factual claim should be supported by a source.
8. Return only the answer to the financial question.
""".strip()

        user_prompt = f"""
QUESTION:
{query}

SOURCES:
{context}

Answer the question using only the sources above.
""".strip()

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
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            },
        )

        if response.status_code == 402:
            raise RuntimeError(
                "OpenRouter rejected the request "
                "because the selected model requires "
                "available credits or billing."
            )

        if response.status_code == 429:
            raise RuntimeError(
                "OpenRouter rate limit exceeded."
            )

        response.raise_for_status()

        data = response.json()

        # ---------------------------------
        # Temporary debugging
        # ---------------------------------

        print(
            "\n===== OPENROUTER DEBUG ====="
        )

        print(
            "Requested model:",
            self.model,
        )

        print(
            "Actual model:",
            data.get("model"),
        )

        print(
            "Finish reason:",
            data.get(
                "choices",
                [{}],
            )[0].get(
                "finish_reason"
            ),
        )

        print(
            "Raw message:",
            data.get(
                "choices",
                [{}],
            )[0].get(
                "message"
            ),
        )

        print(
            "============================\n"
        )

        # ---------------------------------
        # Extract answer
        # ---------------------------------

        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError(
                "OpenRouter returned no choices."
            )

        message = choices[0].get(
            "message",
            {}
        )

        content = message.get("content")

        if not content:
            raise RuntimeError(
                "OpenRouter returned an empty "
                "assistant response."
            )

        return content.strip()