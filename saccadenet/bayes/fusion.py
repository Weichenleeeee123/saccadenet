"""Model B: keep the highest-quality observation per candidate."""

import math

import numpy as np


class FusionB:
    def __init__(self) -> None:
        self._quality: dict[int, float] = {}
        self._llr: dict[int, float] = {}

    def update(self, candidate_id: int, x: float, dprime: float) -> None:
        if not math.isfinite(x) or not math.isfinite(dprime) or dprime < 0:
            raise ValueError("evidence and dprime must be finite; dprime must be nonnegative")
        quality = dprime * dprime
        self.update_llr(candidate_id, quality * (x - 0.5), quality=quality)

    def update_llr(self, candidate_id: int, llr: float, *, quality: float) -> None:
        if not math.isfinite(llr) or not math.isfinite(quality) or quality < 0:
            raise ValueError("llr and quality must be finite and quality nonnegative")
        if quality > self._quality.get(candidate_id, -1.0):
            self._quality[candidate_id] = quality
            self._llr[candidate_id] = llr

    def reset(self, candidate_id: int) -> None:
        self._quality[candidate_id] = 0.0
        self._llr[candidate_id] = 0.0

    def llr(self) -> dict[int, float]:
        return dict(self._llr)

    def quality(self, candidate_id: int) -> float:
        return self._quality.get(candidate_id, 0.0)

    def gain(self, candidate_id: int, dprime: float) -> float:
        return max(0.0, dprime * dprime - self.quality(candidate_id))

    def posterior(self, candidate_ids: list[int]) -> np.ndarray:
        if not candidate_ids:
            return np.empty(0, dtype=np.float64)
        logits = np.asarray([self._llr.get(candidate_id, 0.0) for candidate_id in candidate_ids], dtype=np.float64)
        if not np.all(np.isfinite(logits)):
            raise ValueError("nonfinite candidate log likelihood")
        weights = np.exp(logits - logits.max())
        return weights / weights.sum()
