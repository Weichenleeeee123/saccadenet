"""Bright-card candidate discovery from retina output alone."""

from dataclasses import dataclass

import cv2
import numpy as np

from saccadenet.contracts import Candidate, RetinaOut


@dataclass(frozen=True)
class Detection:
    xy: tuple[float, float]
    score: float
    localization_quality: float | None = None


def _components(mask: np.ndarray, col_range: tuple[int, int] | None = None):
    """Yield (rows, cols, area) per component in row-major pixel order, as np.where(labels == index) would.

    With col_range=(low, high), components whose bounding box misses columns [low, high) are skipped.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        left, top, box_w, box_h, area = (int(value) for value in stats[index, :5])
        if col_range is not None and (left + box_w <= col_range[0] or left >= col_range[1]):
            continue
        rows, cols = np.nonzero(labels[top : top + box_h, left : left + box_w] == index)
        yield rows + top, cols + left, area


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
        span_x = float(cols.max() - cols.min())
        span_y = float(rows.max() - rows.min())
        quality = max(0.0, min(span_x, span_y) - 0.5 * abs(span_x - span_y))
        result.append(Detection((cx, cy), float(fovea_brightness[rows, cols].mean() / 255), quality))

    brightness = retina.logpolar.astype(np.float32).mean(axis=2)
    bright = (brightness >= brightness_threshold) & retina.logpolar_valid
    sectors = bright.shape[1]
    wrapped = np.concatenate((bright, bright, bright), axis=1)
    for rows, cols, area in _components(wrapped, (sectors, 2 * sectors)):
        if area < 1 or not np.any((cols >= sectors) & (cols < 2 * sectors)):
            continue
        # Deduplicate (ring, sector) pairs; np.nonzero returns them in the same sorted order as np.unique(axis=0).
        first_row = int(rows.min())
        seen = np.zeros((int(rows.max()) - first_row + 1, sectors), dtype=bool)
        seen[rows - first_row, cols % sectors] = True
        unique_rows, unique_cols = np.nonzero(seen)
        unique = np.stack((unique_rows + first_row, unique_cols), axis=1)
        sample_locations = retina.sample_xy[unique[:, 0], unique[:, 1]]
        low = np.min(sample_locations, axis=0)
        high = np.max(sample_locations, axis=0)
        position = (low + high) / 2
        span_x, span_y = map(float, high - low)
        quality = max(0.0, min(span_x, span_y) - 0.5 * abs(span_x - span_y))
        intensity = brightness[unique[:, 0], unique[:, 1]].mean()
        result.append(Detection((float(position[0]), float(position[1])), float(intensity / 255), quality))
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
        for detection in sorted(detections, key=lambda item: (-(item.localization_quality if item.localization_quality is not None else item.score), -item.score)):
            quality = detection.localization_quality if detection.localization_quality is not None else detection.score
            nearest = min(
                self._candidates,
                key=lambda candidate: np.linalg.norm(np.asarray(candidate.xy) - detection.xy),
                default=None,
            )
            distance = float(np.linalg.norm(np.asarray(nearest.xy) - detection.xy)) if nearest else float("inf")
            if nearest is None or distance > self.merge_radius:
                self._candidates.append(Candidate(self._next_id, detection.xy, detection.score, step, step, quality))
                self._next_id += 1
                continue
            if quality > nearest.localization_quality or (quality == nearest.localization_quality and detection.score > nearest.detector_score):
                if distance > self.reset_distance:
                    reset_ids.add(nearest.stable_id)
                nearest.xy = detection.xy
                nearest.detector_score = detection.score
                nearest.localization_quality = quality
            nearest.last_seen = step
        return self.candidates, reset_ids
