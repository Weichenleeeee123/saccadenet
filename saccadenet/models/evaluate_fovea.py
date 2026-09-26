"""One-shot held-out 11-class foveal patch evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from saccadenet.data.mnist_bank import MnistBank
from saccadenet.models.fovea import FoveaNet
from saccadenet.models.train_fovea import FoveaPatchDataset


def classification_metrics(confusion: np.ndarray) -> dict:
    if confusion.shape != (11, 11) or np.any(confusion < 0) or confusion.sum() == 0:
        raise ValueError("expected nonempty 11x11 confusion matrix")
    true_count = confusion.sum(axis=1)
    recall = np.divide(np.diag(confusion), true_count, out=np.zeros(11), where=true_count > 0)
    clear_count = true_count[:10].sum()
    return {
        "overall_accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_recall_11": float(recall.mean()),
        "clear_digit_accuracy": float(np.trace(confusion[:10, :10]) / clear_count) if clear_count else float("nan"),
        "blank_false_positive_rate": float(confusion[10, :10].sum() / true_count[10]) if true_count[10] else float("nan"),
        "per_class_recall": recall.tolist(),
        "per_class_count": true_count.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=31000)
    args = parser.parse_args()
    if args.samples <= 0 or not 31000 <= args.seed < 32000:
        raise ValueError("use positive samples and registered held-out seed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    network = FoveaNet().to(device).eval()
    network.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    dataset = FoveaPatchDataset(MnistBank.from_split("final_test"), samples=args.samples, seed=args.seed, blank_fraction=0.1, max_shift=4, degrade_probability=0)
    loader = DataLoader(dataset, batch_size=128, num_workers=0)
    confusion = np.zeros((11, 11), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = network(images.to(device)).argmax(dim=1).cpu().numpy()
            np.add.at(confusion, (labels.numpy(), predictions), 1)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("reports/fovea") / f"heldout-{stamp}-{checkpoint_hash[:8]}.json"
    payload = {"checkpoint_sha256": checkpoint_hash, "split": "MNIST official test", "seed": args.seed, "samples": args.samples, "data_pipeline": "FoveaPatchDataset 96px, digit 48px, max shift 4, no blur, 10% blank", "confusion": confusion.tolist(), **classification_metrics(confusion)}
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps({"output": str(output), "overall_accuracy": payload["overall_accuracy"], "macro_recall_11": payload["macro_recall_11"], "clear_digit_accuracy": payload["clear_digit_accuracy"], "blank_false_positive_rate": payload["blank_false_positive_rate"]}))


if __name__ == "__main__":
    main()
