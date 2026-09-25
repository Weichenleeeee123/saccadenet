"""Bright-card candidate discovery from retina output alone."""

from dataclasses import dataclass

import cv2
import numpy as np

from saccadenet.contracts import Candidate, RetinaOut


@dataclass(frozen=True)
class Detection:
    xy: tuple[float, float]
    score: float


def _components(mask: np.ndarray):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        rows, cols = np.where(labels == index)
        yield rows, cols, int(stats[index, cv2.CC_STAT_AREA])


def detect_on_retina(retina: RetinaOut, *, brightness_threshold: float = 195.0) -> list[Detection]:
    """Locate bright regions; no canvas, pyramid, or ground truth is accepted."""
    result: list[Detection] = []
    fovea_brightness = retina.fovea.astype(np.float32).mean(axis=2)
    fovea_mask = (fovea_brightness >= brightness_threshold) & retina.fovea_valid
    for rows, cols, area in _components(fovea_mask):
        if area < 16:
            continue
        cx = retina.fixation_xy[0] + float(cols.mean()) - retina.fovea.shape[1] / 2
        cy = retina.fixation_xy[1] + float(rows.mean()) - retina.fovea.shape[0] / 2
        result.append(Detection((cx, cy), float(fovea_brightness[rows, cols].mean() / 255)))

    brightness = retina.logpolar.astype(np.float32).mean(axis=2)
    bright = (brightness >= brightness_threshold) & retina.logpolar_valid
    sectors = bright.shape[1]
    wrapped = np.concatenate((bright, bright, bright), axis=1)
    for rows, cols, area in _components(wrapped):
        if area < 1 or not np.any((cols >= sectors) & (cols < 2 * sectors)):
            continue
        unique = np.unique(np.stack((rows, cols % sectors), axis=1), axis=0)
        sample_locations = retina.sample_xy[unique[:, 0], unique[:, 1]]
        position = np.mean(sample_locations, axis=0)
        intensity = brightness[unique[:, 0], unique[:, 1]].mean()
        result.append(Detection((float(position[0]), float(position[1])), float(intensity / 255)))
    return result


class CandidateTracker:
    def __init__(self, merge_radius: float = 144, reset_distance: float = 8):
        self.merge_radius = merge_radius
        self.reset_distance = reset_distance
        self._candidates: list[Candidate] = []
        self._next_id = 0

    @property
    def candidates(self) -> list[Candidate]:
        return list(self._candidates)

    def update(self, detections: list[Detection], step: int) -> tuple[list[Candidate], set[int]]:
        reset_ids: set[int] = set()
        for detection in sorted(detections, key=lambda item: -item.score):
            nearest = min(
                self._candidates,
                key=lambda candidate: np.linalg.norm(np.asarray(candidate.xy) - detection.xy),
                default=None,
            )
            distance = float(np.linalg.norm(np.asarray(nearest.xy) - detection.xy)) if nearest else float("inf")
            if nearest is None or distance > self.merge_radius:
                self._candidates.append(Candidate(self._next_id, detection.xy, detection.score, step, step))
                self._next_id += 1
                continue
            if detection.score > nearest.detector_score:
                if distance > self.reset_distance:
                    reset_ids.add(nearest.stable_id)
                nearest.xy = detection.xy
                nearest.detector_score = detection.score
            nearest.last_seen = step
        return self.candidates, reset_ids
