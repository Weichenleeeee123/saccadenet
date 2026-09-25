"""Measure in-memory synthetic canvas generation and peak process RAM."""

import json
import threading
import time

import numpy as np
import psutil

from saccadenet.config import EpisodeConfig, RESOLUTIONS
from saccadenet.data.canvas import CANVAS_VERSION, make_canvas


class SyntheticDigitBank:
    def sample(self, label, rng):
        return np.full((28, 28), 255, dtype=np.uint8), int(rng.integers(0, 1_000_000))


def measure(width: int, height: int) -> dict:
    process = psutil.Process()
    stop = threading.Event()
    peak = [process.memory_info().rss]

    def watch() -> None:
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        image, truth = make_canvas(EpisodeConfig(width=width, height=height), 123, SyntheticDigitBank())
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        return {
            "resolution": [width, height],
            "seconds": round(elapsed, 3),
            "peak_rss_gib": round(peak[0] / 2**30, 3),
            "image_bytes": image.nbytes,
            "cards": len(truth.digits),
            "canvas_version": CANVAS_VERSION,
        }
    finally:
        stop.set()
        thread.join()


if __name__ == "__main__":
    for width, height in RESOLUTIONS:
        print(json.dumps(measure(width, height), sort_keys=True), flush=True)
