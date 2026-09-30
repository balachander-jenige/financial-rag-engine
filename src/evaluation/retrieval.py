from dataclasses import dataclass

from src.evaluation.dataset import RetrievalExample
from src.retrieval.dense import DenseRetriever


@dataclass
class EvaluationResult:
    example_id: str
    query: str
    expected_text: str

    first_relevant_rank: int | None

    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool

    reciprocal_rank: float


def evaluate_example(
    example: RetrievalExample,
    retriever: DenseRetriever,
) -> EvaluationResult:

    results = retriever.retrieve(
        example.query
    )

    first_relevant_rank = None

    expected = example.expected_text.lower()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        # RetrievalResult now contains
        # the text directly.
        text = result.text.lower()

        if expected in text:
            first_relevant_rank = rank
            break

    reciprocal_rank = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    return EvaluationResult(
        example_id=example.id,
        query=example.query,
        expected_text=example.expected_text,

        first_relevant_rank=first_relevant_rank,

        hit_at_1=(
            first_relevant_rank == 1
        ),

        hit_at_3=(
            first_relevant_rank is not None
            and first_relevant_rank <= 3
        ),

        hit_at_5=(
            first_relevant_rank is not None
            and first_relevant_rank <= 5
        ),

        hit_at_10=(
            first_relevant_rank is not None
            and first_relevant_rank <= 10
        ),

        reciprocal_rank=reciprocal_rank,
    )