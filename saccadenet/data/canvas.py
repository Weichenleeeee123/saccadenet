"""Deterministic, bounded-memory synthetic visual-search canvases."""

from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from saccadenet.config import EpisodeConfig
from saccadenet.contracts import SceneTruth

CANVAS_VERSION = "synthetic-v1"


class DigitBank(Protocol):
    def sample(self, label: int, rng: np.random.Generator) -> tuple[np.ndarray, int]: ...


def _background(width: int, height: int, contrast: float, rng: np.random.Generator) -> Image.Image:
    small_w, small_h = 256, 144
    composed = np.zeros((small_h, small_w), dtype=np.float32)
    for grid_w, grid_h, weight in ((16, 9, 0.50), (32, 18, 0.26), (64, 36, 0.16), (128, 72, 0.08)):
        random_grid = rng.integers(0, 256, (grid_h, grid_w), dtype=np.uint8)
        resized = Image.fromarray(random_grid).resize((small_w, small_h), Image.Resampling.BILINEAR)
        composed += weight * np.asarray(resized, dtype=np.float32)
    composed = np.clip(128 + contrast * (composed - composed.mean()) * 2.0, 0, 255)
    low = Image.fromarray(composed.astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    gray = np.array(low, dtype=np.uint8, copy=True)
    tile = rng.integers(-12, 13, size=(256, 256), dtype=np.int16)
    for y in range(0, height, 256):
        h = min(256, height - y)
        fine = np.tile(tile[:h], (1, (width + 255) // 256))[:, :width]
        gray[y : y + h] = np.clip(gray[y : y + h].astype(np.int16) + contrast * fine, 0, 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def _place_centers(config: EpisodeConfig, rng: np.random.Generator) -> np.ndarray:
    half = config.card_size // 2
    centers: list[tuple[int, int]] = []
    for _ in range(10_000):
        if len(centers) == config.k:
            break
        x = int(rng.integers(half, config.width - half + 1))
        y = int(rng.integers(half, config.height - half + 1))
        if all((x - old_x) ** 2 + (y - old_y) ** 2 >= config.card_min_distance ** 2 for old_x, old_y in centers):
            centers.append((x, y))
    if len(centers) != config.k:
        raise ValueError(f"placement_failed: placed {len(centers)} of {config.k} cards")
    return np.asarray(centers, dtype=np.int32)


def make_canvas(
    config: EpisodeConfig,
    seed: int,
    digits: DigitBank,
) -> tuple[np.ndarray, SceneTruth]:
    """Generate one scene; ground truth is returned separately from the image."""
    streams = np.random.SeedSequence(seed).spawn(4)
    background_rng, layout_rng, label_rng, jitter_rng = (np.random.default_rng(item) for item in streams)
    centers = _place_centers(config, layout_rng)
    target_index = int(label_rng.integers(0, config.k))
    labels = [config.query if index == target_index else int(label_rng.choice([digit for digit in range(10) if digit != config.query])) for index in range(config.k)]
    digit_ids: list[int] = []
    image = _background(config.width, config.height, config.background_contrast, background_rng)
    painter = ImageDraw.Draw(image)
    card_half = config.card_size // 2
    for index, (x, y) in enumerate(centers):
        if config.cards_on:
            shade = 220 if index == target_index else int(220 - 32 * config.color_similarity)
            painter.rectangle((int(x) - card_half, int(y) - card_half, int(x) + card_half - 1, int(y) + card_half - 1), fill=(shade, shade, shade))
        digit_array, digit_id = digits.sample(labels[index], label_rng)
        digit_ids.append(int(digit_id))
        if digit_array.shape != (28, 28):
            raise ValueError("digit bank must return a 28x28 image")
        digit = Image.fromarray(255 - digit_array.astype(np.uint8)).resize((config.digit_size, config.digit_size), Image.Resampling.BILINEAR)
        offset_x = int(jitter_rng.integers(-4, 5))
        offset_y = int(jitter_rng.integers(-4, 5))
        left = int(x) - config.digit_size // 2 + offset_x
        top = int(y) - config.digit_size // 2 + offset_y
        image.paste(Image.merge("RGB", (digit, digit, digit)), (left, top))
    truth = SceneTruth(centers_xy=centers, digits=tuple(labels), digit_ids=tuple(digit_ids), target_index=target_index)
    return np.asarray(image), truth
