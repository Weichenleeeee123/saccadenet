"""Validated experiment configuration and stable run identity."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from typing import Any, Mapping


RESOLUTIONS = ((1920, 1080), (3840, 2160), (7680, 4320), (15360, 8640))


@dataclass(frozen=True)
class EpisodeConfig:
    width: int = 1920
    height: int = 1080
    k: int = 12
    query: int = 0
    tau: float = 0.95
    t_max: int = 40
    fovea_size: int = 96
    sectors: int = 128
    dprime_min: float = 0.1
    card_size: int = 192
    digit_size: int = 48
    card_min_distance: int = 288
    cards_on: bool = True
    color_similarity: float = 0.0
    background_contrast: float = 0.3

    def __post_init__(self) -> None:
        if (self.width, self.height) not in RESOLUTIONS:
            raise ValueError("resolution must be one of 1080p, 4K, 8K, or 16K")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not 0 <= self.query <= 9:
            raise ValueError("query must be a digit from 0 to 9")
        if not 0 < self.tau < 1:
            raise ValueError("tau must lie strictly between 0 and 1")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")
        if self.fovea_size <= 0 or self.fovea_size % 2:
            raise ValueError("fovea_size must be a positive even integer")
        if self.sectors < 8:
            raise ValueError("sectors must be at least 8")
        if self.dprime_min < 0:
            raise ValueError("dprime_min must be nonnegative")
        if self.card_size <= 0 or self.digit_size > self.card_size:
            raise ValueError("card_size and digit_size are inconsistent")
        if self.card_min_distance < self.card_size:
            raise ValueError("card_min_distance must not be smaller than card_size")
        if not 0 <= self.color_similarity <= 1:
            raise ValueError("color_similarity must lie in [0, 1]")
        if not 0 <= self.background_contrast <= 1:
            raise ValueError("background_contrast must lie in [0, 1]")


def config_from_mapping(values: Mapping[str, Any]) -> EpisodeConfig:
    allowed = {field.name for field in fields(EpisodeConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    return EpisodeConfig(**values)


def config_hash(config: EpisodeConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

