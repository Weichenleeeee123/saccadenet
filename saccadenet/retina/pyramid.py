"""Anti-aliased, judgment-free image pyramid."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Pyramid:
    levels: tuple[np.ndarray, ...]
    add_count: int


def build_pyramid(image: np.ndarray) -> Pyramid:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be HxWx3 uint8")
    levels = [image]
    add_count = 0
    while min(levels[-1].shape[:2]) >= 64:
        current = levels[-1]
        next_w = (current.shape[1] + 1) // 2
        next_h = (current.shape[0] + 1) // 2
        # INTER_AREA reduces even dimensions with 2x2 region averages.
        # Fractional edge regions on odd dimensions remain anti-aliased.
        reduced = cv2.resize(current, (next_w, next_h), interpolation=cv2.INTER_AREA)
        levels.append(reduced)
        add_count += int(reduced.size) * 3
    return Pyramid(levels=tuple(levels), add_count=add_count)
