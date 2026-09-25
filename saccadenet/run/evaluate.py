"""Ground-truth access lives only in this evaluation module."""

from dataclasses import dataclass
import math

from saccadenet.contracts import SceneTruth


@dataclass(frozen=True)
class AnswerEvaluation:
    target_hit: bool
    matched_card_index: int | None
    localization_error: float | None


def evaluate_answer(
    answer_xy: tuple[float, float] | None, truth: SceneTruth, match_radius: float = 48
) -> AnswerEvaluation:
    if answer_xy is None:
        return AnswerEvaluation(False, None, None)
    distances = [math.dist(answer_xy, tuple(center)) for center in truth.centers_xy]
    nearest = min(range(len(distances)), key=distances.__getitem__)
    if distances[nearest] > match_radius:
        return AnswerEvaluation(False, None, distances[nearest])
    return AnswerEvaluation(nearest == truth.target_index, nearest, distances[nearest])

