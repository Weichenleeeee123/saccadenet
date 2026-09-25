"""ELM-style precision gain, MAP baseline, and fixed exploration fallback."""

from collections.abc import Callable, Iterable
import math

import numpy as np

from saccadenet.bayes.fusion import FusionB
from saccadenet.contracts import Candidate


def fixation_candidates(
    candidates: list[Candidate], *, e_half: float, canvas_shape: tuple[int, int]
) -> list[tuple[float, float]]:
    height, width = canvas_shape
    options = {tuple(candidate.xy) for candidate in candidates}
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if math.dist(left.xy, right.xy) < 2 * e_half:
                midpoint = ((left.xy[0] + right.xy[0]) / 2, (left.xy[1] + right.xy[1]) / 2)
                if 0 <= midpoint[0] < width and 0 <= midpoint[1] < height:
                    options.add(midpoint)
    return sorted(options)


def choose_fixation(
    posterior: np.ndarray,
    candidates: list[Candidate],
    fusion: FusionB,
    omega: Iterable[tuple[float, float]],
    *,
    current: tuple[float, float],
    dprime: Callable[[float], float],
    strategy: str = "gain",
    epsilon: float = 1e-4,
) -> tuple[float, float] | None:
    if len(posterior) != len(candidates):
        raise ValueError("posterior and candidates differ in length")
    if not candidates:
        return None
    if strategy == "map":
        return candidates[int(np.argmax(posterior))].xy
    if strategy != "gain":
        raise ValueError(f"unknown strategy: {strategy}")
    options = list(omega)
    if not options:
        return None
    gains = []
    for xy in options:
        gain = sum(
            float(probability) * fusion.gain(candidate.stable_id, dprime(math.dist(xy, candidate.xy)))
            for probability, candidate in zip(posterior, candidates)
            if probability > epsilon
        )
        gains.append(gain)
    best = max(gains)
    if best <= 1e-12:
        return None
    tied = [xy for xy, gain in zip(options, gains) if abs(gain - best) <= 1e-12]
    return min(tied, key=lambda xy: (math.dist(current, xy), xy[0], xy[1]))


def exploration_anchor(
    width: int,
    height: int,
    *,
    grid: int,
    visited: set[tuple[float, float]],
    current: tuple[float, float],
) -> tuple[float, float] | None:
    if grid <= 0:
        raise ValueError("grid must be positive")
    remaining = [
        ((column + 0.5) * width / grid, (row + 0.5) * height / grid)
        for row in range(grid)
        for column in range(grid)
        if ((column + 0.5) * width / grid, (row + 0.5) * height / grid) not in visited
    ]
    return min(remaining, key=lambda xy: (math.dist(current, xy), xy[0], xy[1])) if remaining else None


def should_stop(
    posterior: np.ndarray,
    *,
    discovered: int,
    expected_k: int,
    tau: float,
    has_evidence: bool,
) -> bool:
    return bool(has_evidence and discovered >= expected_k and posterior.size and np.max(posterior) >= tau)
