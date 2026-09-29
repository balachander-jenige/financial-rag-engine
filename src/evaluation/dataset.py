import json
from pathlib import Path

from pydantic import BaseModel


class RetrievalExample(BaseModel):
    id: str
    query: str

    company: str
    fiscal_year: int

    expected_text: str


def load_retrieval_dataset(
    path: Path = Path(
        "data/evaluation/retrieval_questions.json"
    ),
) -> list[RetrievalExample]:

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    return [
        RetrievalExample.model_validate(item)
        for item in data
    ]