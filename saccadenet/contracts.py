"""Data boundaries used by the sensor, inference loop, and evaluator."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SceneTruth:
    centers_xy: np.ndarray
    digits: tuple[int, ...]
    digit_ids: tuple[int, ...]
    target_index: int


@dataclass(frozen=True)
class EpisodeInput:
    query: int
    width: int
    height: int
    expected_k: int


@dataclass(frozen=True)
class RetinaOut:
    fovea: np.ndarray
    logpolar: np.ndarray
    fovea_valid: np.ndarray
    logpolar_valid: np.ndarray
    sample_xy: np.ndarray
    fixation_xy: tuple[float, float]
    canvas_shape: tuple[int, int]
    sensing_flops: int


@dataclass
class Candidate:
    stable_id: int
    xy: tuple[float, float]
    detector_score: float
    first_seen: int
    last_seen: int


@dataclass(frozen=True)
class Observation:
    candidate_id: int
    evidence: float
    dprime: float
    valid_fraction: float
    mode: str = "gaussian_x"


@dataclass
class EpisodeLog:
    answer_xy: tuple[float, float] | None
    reason: str
    steps: int
    trace: list[dict] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class Sensor(Protocol):
    def observe(self, fixation_xy: tuple[float, float]) -> RetinaOut: ...
