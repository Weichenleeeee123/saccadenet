"""Log-polar retina with eccentricity-dependent anti-aliased sampling."""

import math

import cv2
import numpy as np

from saccadenet.contracts import RetinaOut
from saccadenet.retina.pyramid import Pyramid


def ring_count(canvas_shape: tuple[int, int], fovea_size: int = 96, sectors: int = 128) -> int:
    height, width = canvas_shape
    diagonal = math.hypot(width, height)
    # j starts at zero. Include the endpoint ring whose center reaches Diag.
    return math.ceil(math.log(diagonal / (fovea_size / 2)) / (2 * math.pi / sectors)) + 1


def _remap(image: np.ndarray, xy: np.ndarray, scale: int = 1) -> np.ndarray:
    mapped_x = ((xy[..., 0] + 0.5) / scale - 0.5).astype(np.float32)
    mapped_y = ((xy[..., 1] + 0.5) / scale - 0.5).astype(np.float32)
    return cv2.remap(image, mapped_x, mapped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sample_retina(
    pyramid: Pyramid,
    fixation_xy: tuple[float, float],
    *,
    fovea_size: int = 96,
    sectors: int = 128,
) -> RetinaOut:
    height, width = pyramid.levels[0].shape[:2]
    radius = fovea_size / 2
    row = np.arange(fovea_size, dtype=np.float32) - radius
    grid_x, grid_y = np.meshgrid(row + fixation_xy[0], row + fixation_xy[1])
    fovea_xy = np.stack((grid_x, grid_y), axis=-1)
    fovea_valid = (grid_x >= 0) & (grid_x < width) & (grid_y >= 0) & (grid_y < height)
    fovea = _remap(pyramid.levels[0], fovea_xy)

    n_rings = ring_count((height, width), fovea_size, sectors)
    delta = 2 * math.pi / sectors
    radii = radius * np.exp(np.arange(n_rings, dtype=np.float64) * delta)
    theta = np.arange(sectors, dtype=np.float64) * delta
    sample_x = fixation_xy[0] + radii[:, None] * np.cos(theta)[None, :]
    sample_y = fixation_xy[1] + radii[:, None] * np.sin(theta)[None, :]
    sample_xy = np.stack((sample_x, sample_y), axis=-1).astype(np.float32)
    logpolar_valid = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    logpolar = np.empty((n_rings, sectors, 3), dtype=np.uint8)
    top_level = len(pyramid.levels) - 1
    levels = [min(max(round(math.log2(max(float(r) * delta, 1.0))), 0), top_level) for r in radii]
    # Levels never decrease with radius, so each level is one contiguous block of rings: one remap per block.
    start = 0
    for stop in range(1, n_rings + 1):
        if stop == n_rings or levels[stop] != levels[start]:
            level = levels[start]
            logpolar[start:stop] = _remap(pyramid.levels[level], sample_xy[start:stop], 2**level)
            start = stop
    sample_count = fovea_size * fovea_size + n_rings * sectors
    return RetinaOut(
        fovea=fovea,
        logpolar=logpolar,
        fovea_valid=fovea_valid,
        logpolar_valid=logpolar_valid,
        sample_xy=sample_xy,
        fixation_xy=fixation_xy,
        canvas_shape=(height, width),
        sensing_flops=pyramid.add_count + sample_count * 3 * 8,
    )
