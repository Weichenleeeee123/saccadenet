"""Synthetic foveal training data and restorable training state."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import BinaryIO

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml

from saccadenet.data.mnist_bank import MnistBank
from saccadenet.models.fovea import FoveaNet


class FoveaPatchDataset(Dataset):
    def __init__(
        self,
        bank,
        *,
        samples: int,
        seed: int,
        blank_fraction: float = 0.1,
        max_shift: int = 20,
        degrade_probability: float = 0.3,
        scale_augment_probability: float = 0.0,
        digit_sizes: tuple[int, ...] = (48,),
    ) -> None:
        if samples <= 0 or not 0 <= blank_fraction <= 1 or max_shift < 0:
            raise ValueError("invalid patch dataset settings")
        if not 0 <= scale_augment_probability <= 1 or not digit_sizes or any(not 0 < size <= 48 for size in digit_sizes):
            raise ValueError("invalid digit size augmentation")
        self.bank = bank
        self.samples = samples
        self.seed = seed
        self.blank_fraction = blank_fraction
        self.max_shift = max_shift
        self.degrade_probability = degrade_probability
        self.scale_augment_probability = scale_augment_probability
        self.digit_sizes = tuple(digit_sizes)
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))
        blank = rng.random() < self.blank_fraction
        label = 10 if blank else int(rng.integers(0, 10))
        background = 128 if blank and rng.random() < 0.5 else 220
        gray = np.clip(background + rng.normal(0, 4, (96, 96)), 0, 255).astype(np.uint8)
        if not blank:
            digit, _ = self.bank.sample(label, rng)
            size = int(rng.choice(self.digit_sizes)) if rng.random() < self.scale_augment_probability else 48
            strokes = cv2.resize(
                digit, (size, size), interpolation=cv2.INTER_AREA if size < 28 else cv2.INTER_CUBIC
            ).astype(np.float32)
            shift_x = int(rng.integers(-self.max_shift, self.max_shift + 1))
            shift_y = int(rng.integers(-self.max_shift, self.max_shift + 1))
            left, top = 48 - size // 2 + shift_x, 48 - size // 2 + shift_y
            gray[top : top + size, left : left + size] = np.clip(
                gray[top : top + size, left : left + size].astype(np.float32) - strokes * (190 / 255),
                0,
                255,
            ).astype(np.uint8)
        if rng.random() < self.degrade_probability:
            factor = int(rng.choice((2, 4)))
            gray = cv2.resize(
                cv2.resize(gray, (96 // factor, 96 // factor), interpolation=cv2.INTER_AREA),
                (96, 96),
                interpolation=cv2.INTER_LINEAR,
            )
        rgb = np.repeat(gray[None, :, :], 3, axis=0).copy()
        return torch.from_numpy(rgb).float().div_(255), label


def save_training_state(
    destination: BinaryIO,
    network: FoveaNet,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config_hash: str,
) -> None:
    torch.save(
        {
            "model": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config_hash": config_hash,
            "torch_rng": torch.get_rng_state(),
        },
        destination,
    )


def load_training_state(
    source: BinaryIO,
    network: FoveaNet,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, str]:
    state = torch.load(source, map_location="cpu", weights_only=True)
    network.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    return int(state["epoch"]), str(state["config_hash"])


def train_one_batch(
    network: FoveaNet,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    network.train()
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(network(images), labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def resolve_end_epoch(*, config_epochs: int, start_epoch: int, additional_epochs: int) -> int:
    if config_epochs <= 0 or additional_epochs < 0:
        raise ValueError("additional_epochs must be nonnegative and config_epochs positive")
    return start_epoch + additional_epochs if additional_epochs else config_epochs


@torch.inference_mode()
def evaluate(network: FoveaNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    network.eval()
    total = correct = clear_total = clear_correct = 0
    for images, labels in loader:
        predictions = network(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
        total += len(labels)
        correct += int((predictions == labels).sum())
        clear = labels != 10
        clear_total += int(clear.sum())
        clear_correct += int(((predictions == labels) & clear).sum())
    return {"all_accuracy": correct / total, "clear_digit_accuracy": clear_correct / clear_total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--additional-epochs", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.resume is not None and args.init_weights is not None:
        raise ValueError("--resume and --init-weights are mutually exclusive")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(int(cfg["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bank = MnistBank.from_split("train")
    validation_bank = MnistBank.from_split("development")
    train_set = FoveaPatchDataset(
        train_bank,
        samples=min(512, int(cfg["train_samples"])) if args.smoke else int(cfg["train_samples"]),
        seed=int(cfg["seed"]),
        blank_fraction=float(cfg["blank_fraction"]),
        max_shift=int(cfg["max_shift"]),
        degrade_probability=float(cfg["degrade_probability"]),
        scale_augment_probability=float(cfg.get("scale_augment_probability", 0)),
        digit_sizes=tuple(cfg.get("digit_sizes", (48,))),
    )
    validation_set = FoveaPatchDataset(
        validation_bank,
        samples=min(256, int(cfg["validation_samples"])) if args.smoke else int(cfg["validation_samples"]),
        seed=int(cfg["seed"]) + 1,
        blank_fraction=float(cfg["blank_fraction"]),
        max_shift=4,
        degrade_probability=0,
    )
    validation_loader = DataLoader(validation_set, batch_size=int(cfg["batch_size"]), num_workers=0)
    network = FoveaNet().to(device)
    if args.init_weights is not None:
        network.load_state_dict(torch.load(args.init_weights, map_location="cpu", weights_only=True)["model"])
    optimizer = torch.optim.Adam(network.parameters(), lr=float(cfg["learning_rate"]))
    start_epoch = 0
    if args.resume is not None:
        with args.resume.open("rb") as file:
            previous_epoch, previous_hash = load_training_state(file, network, optimizer)
        if previous_hash != digest:
            raise ValueError("checkpoint configuration hash does not match")
        start_epoch = previous_epoch + 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}-{digest[:8]}{'-smoke' if args.smoke else ''}"
    run_dir = Path("checkpoints/fovea") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = Path("reports/fovea") / f"training-{run_id}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    final_epoch = 1 if args.smoke else resolve_end_epoch(
        config_epochs=int(cfg["epochs"]), start_epoch=start_epoch, additional_epochs=args.additional_epochs
    )
    if final_epoch <= start_epoch:
        raise ValueError("no training epochs scheduled; use --additional-epochs when resuming a complete run")
    manifest = report.parent / f"manifest-{run_id}.json"
    with manifest.open("x", encoding="utf-8") as file:
        json.dump(
            {
                "config": cfg,
                "config_hash": digest,
                "device": str(device),
                "resume_from": str(args.resume) if args.resume else None,
                "init_weights_from": str(args.init_weights) if args.init_weights else None,
                "start_epoch": start_epoch,
                "end_epoch": final_epoch,
                "train_source": "MNIST/train/0:50000",
                "validation_source": "MNIST/train/55000:60000",
            },
            file,
            indent=2,
        )
    with report.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("epoch", "train_loss", "validation_accuracy", "clear_digit_accuracy", "seconds", "checkpoint"),
        )
        writer.writeheader()
        for epoch in range(start_epoch, final_epoch):
            train_set.epoch = epoch
            generator = torch.Generator().manual_seed(int(cfg["seed"]) + epoch)
            loader = DataLoader(train_set, batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator, num_workers=0)
            started = time.perf_counter()
            losses = []
            for images, labels in loader:
                losses.append(train_one_batch(network, optimizer, images, labels, device=device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            metrics = evaluate(network, validation_loader, device)
            checkpoint = run_dir / f"epoch-{epoch + 1:03d}.pt"
            with checkpoint.open("xb") as destination:
                save_training_state(destination, network, optimizer, epoch=epoch, config_hash=digest)
            row = {
                "epoch": epoch + 1,
                "train_loss": sum(losses) / len(losses),
                "validation_accuracy": metrics["all_accuracy"],
                "clear_digit_accuracy": metrics["clear_digit_accuracy"],
                "seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint),
            }
            writer.writerow(row)
            output.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
