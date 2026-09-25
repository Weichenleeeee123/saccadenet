"""Measure minimum rendered digit size for a fixed foveal classifier."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from saccadenet.data.mnist_bank import MnistBank
from saccadenet.models.fovea import FoveaNet


SIZES = (4, 6, 8, 10, 12, 16, 24, 32, 48)


def render_digit_at_size(
    source: np.ndarray, size: int, *, offset: tuple[int, int] = (0, 0)
) -> np.ndarray:
    if source.shape != (28, 28) or not 0 < size <= 48:
        raise ValueError("source must be 28x28 and size in 1..48")
    patch = np.full((96, 96), 220, dtype=np.uint8)
    strokes = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA if size < 28 else cv2.INTER_CUBIC)
    left, top = 48 - size // 2 + offset[0], 48 - size // 2 + offset[1]
    patch[top : top + size, left : left + size] = np.clip(
        220 - strokes.astype(np.float32) * (190 / 255), 0, 255
    ).astype(np.uint8)
    return np.repeat(patch[:, :, None], 3, axis=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = FoveaNet().to(device).eval()
    network.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    bank = MnistBank.from_split("calibration")
    rng = np.random.default_rng(501)
    examples = []
    for _ in range(args.samples):
        label = int(rng.integers(0, 10))
        source, digit_id = bank.sample(label, rng)
        offset = (int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
        examples.append((label, source, digit_id, offset))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"s-min-{stamp}-{checkpoint_hash[:8]}"
    out_dir = Path("reports/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in SIZES:
        correct = 0
        for start in range(0, len(examples), 128):
            batch = examples[start : start + 128]
            pixels = np.stack([render_digit_at_size(source, size, offset=offset) for _, source, _, offset in batch])
            tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
            with torch.inference_mode():
                prediction = network(tensor).argmax(dim=1).cpu().numpy()
            correct += int(np.sum(prediction == np.asarray([label for label, _, _, _ in batch])))
        row = {"digit_size": size, "n": len(examples), "correct": correct, "accuracy": correct / len(examples)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    viable = [row["digit_size"] for row in rows if row["accuracy"] >= 0.9]
    s_min = min(viable) if viable else None
    with (out_dir / f"{name}.json").open("x", encoding="utf-8") as output:
        json.dump(
            {
                "checkpoint_sha256": checkpoint_hash,
                "source": "MNIST/train/calibration/50000:55000",
                "seed": 501,
                "s_min": s_min,
                "predicted_collapse_width": 48 * 1024 / s_min if s_min else None,
                "threshold_accuracy": 0.9,
                "csv": str(csv_path),
            },
            output,
            indent=2,
        )


if __name__ == "__main__":
    main()
