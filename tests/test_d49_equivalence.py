"""D49: the performance refactor must be bit-identical to the frozen pre-D49 implementations below."""

import math

import cv2
import numpy as np
import pytest

from saccadenet.bayes.calibrate import GaussianCalibrator
from saccadenet.retina.detect import Detection, _components, detect_on_retina
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import _remap, ring_count, sample_retina


# ---- Frozen reference implementations (code as of 40fada5) ----

def reference_logpolar(pyramid, fixation_xy, fovea_size=96, sectors=128):
    height, width = pyramid.levels[0].shape[:2]
    radius = fovea_size / 2
    n_rings = ring_count((height, width), fovea_size, sectors)
    delta = 2 * math.pi / sectors
    radii = radius * np.exp(np.arange(n_rings, dtype=np.float64) * delta)
    theta = np.arange(sectors, dtype=np.float64) * delta
    sample_x = fixation_xy[0] + radii[:, None] * np.cos(theta)[None, :]
    sample_y = fixation_xy[1] + radii[:, None] * np.sin(theta)[None, :]
    sample_xy = np.stack((sample_x, sample_y), axis=-1).astype(np.float32)
    logpolar = np.empty((n_rings, sectors, 3), dtype=np.uint8)
    for index, ring_radius in enumerate(radii):
        spacing = ring_radius * delta
        level = int(np.clip(round(math.log2(max(spacing, 1.0))), 0, len(pyramid.levels) - 1))
        logpolar[index : index + 1] = _remap(pyramid.levels[level], sample_xy[index : index + 1], 2**level)
    return logpolar


def reference_components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        rows, cols = np.where(labels == index)
        yield rows, cols, int(stats[index, cv2.CC_STAT_AREA])


def reference_detect(retina, brightness_threshold=195.0):
    result = []
    fovea_brightness = retina.fovea.astype(np.float32).mean(axis=2)
    fovea_mask = (fovea_brightness >= brightness_threshold) & retina.fovea_valid
    for rows, cols, area in reference_components(fovea_mask):
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
    for rows, cols, area in reference_components(wrapped):
        if area < 1 or not np.any((cols >= sectors) & (cols < 2 * sectors)):
            continue
        unique = np.unique(np.stack((rows, cols % sectors), axis=1), axis=0)
        sample_locations = retina.sample_xy[unique[:, 0], unique[:, 1]]
        low = np.min(sample_locations, axis=0)
        high = np.max(sample_locations, axis=0)
        position = (low + high) / 2
        span_x, span_y = map(float, high - low)
        quality = max(0.0, min(span_x, span_y) - 0.5 * abs(span_x - span_y))
        intensity = brightness[unique[:, 0], unique[:, 1]].mean()
        result.append(Detection((float(position[0]), float(position[1])), float(intensity / 255), quality))
    return result


def reference_index(calibrator, eccentricity):
    return int(np.clip(np.searchsorted(calibrator.edges, eccentricity, side="right") - 1, 0, len(calibrator.mu0) - 1))


# ---- Inputs ----

def card_canvas(width, height, seed, cards=12):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 120, size=(height, width, 3), dtype=np.uint8)
    for _ in range(cards):
        size = int(rng.integers(40, 260))
        x = int(rng.integers(-size // 2, width - size // 2))
        y = int(rng.integers(-size // 2, height - size // 2))
        image[max(y, 0) : y + size, max(x, 0) : x + size] = rng.integers(200, 256, size=3, dtype=np.uint8)
    return image


def fixations(width, height, seed, count=12):
    rng = np.random.default_rng(seed)
    points = [(width / 2, height / 2), (0.0, 0.0), (width - 1.0, height - 1.0), (-50.0, height / 3), (width + 30.0, 17.5)]
    points += [(float(rng.uniform(0, width)), float(rng.uniform(0, height))) for _ in range(count - len(points))]
    return points


CANVASES = [(128, 128), (640, 360), (1920, 1080), (3840, 2160)]


@pytest.mark.parametrize(("width", "height"), CANVASES)
def test_grouped_remap_is_bit_identical(width, height):
    pyramid = build_pyramid(card_canvas(width, height, seed=width))
    for fixation in fixations(width, height, seed=height):
        assert np.array_equal(sample_retina(pyramid, fixation).logpolar, reference_logpolar(pyramid, fixation))


@pytest.mark.parametrize(("width", "height"), CANVASES)
def test_detection_is_bit_identical(width, height):
    pyramid = build_pyramid(card_canvas(width, height, seed=width + 1))
    total = 0
    for fixation in fixations(width, height, seed=height + 1):
        retina = sample_retina(pyramid, fixation)
        new, old = detect_on_retina(retina), reference_detect(retina)
        assert new == old  # dataclass equality: exact float equality on xy, score, quality
        total += len(new)
    assert total > 0


def test_components_match_reference_on_random_masks():
    rng = np.random.default_rng(49)
    for trial in range(200):
        mask = rng.random((int(rng.integers(1, 40)), int(rng.integers(1, 60)))) < rng.uniform(0.05, 0.6)
        new, old = list(_components(mask)), list(reference_components(mask))
        assert len(new) == len(old)
        for (r1, c1, a1), (r0, c0, a0) in zip(new, old):
            assert a1 == a0 and np.array_equal(r1, r0) and np.array_equal(c1, c0) and r1.dtype == r0.dtype


def test_bisect_index_matches_searchsorted():
    calibrator = GaussianCalibrator(
        edges=(0.0, 50.0, 150.0, 400.0, 1000.0, 3000.0),
        mu0=(0.0,) * 5, mu1=(1.0,) * 5, sigma=(1.0,) * 5, count0=(2,) * 5, count1=(2,) * 5,
    )
    probes = [-1.0, 0.0, 1e-9, 49.999, 50.0, 50.0001, 150.0, 399.0, 1000.0, 2999.9, 3000.0, 1e9, math.inf, -math.inf, math.nan]
    probes += list(np.random.default_rng(1).uniform(-100, 4000, size=2000))
    probes += [np.float32(150.0), np.float64(400.0)]
    for value in probes:
        assert calibrator._index(value) == reference_index(calibrator, value), value
