"""Fixed-seed Day-0 policy toy: gain versus MAP under three observation assumptions."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax


LAYOUTS = {
    "sparse": np.array([[0, 0], [800, 0], [0, 800], [800, 800]], dtype=float),
    "dense": np.array([[0, 0], [150, 0], [0, 150], [150, 150]], dtype=float),
}


def dprime(eccentricity: np.ndarray) -> np.ndarray:
    return 4 * np.exp(-eccentricity / 140)


def one_episode(layout: np.ndarray, target: int, *, model: str, strategy: str, seed: int, max_steps: int = 15) -> tuple[int, bool]:
    rng = np.random.default_rng(seed)
    n = len(layout)
    latent = rng.normal(0, 0.2, n) if model == "C" else np.zeros(n)
    llr = np.zeros(n)
    best_quality = np.zeros(n)
    weighted_score = np.zeros(n)
    independent_precision = np.zeros(n)
    posterior = np.full(n, 1 / n)
    current = layout.mean(axis=0)
    omega = np.unique(np.concatenate((layout, (layout[:, None] + layout[None, :]).reshape(-1, 2) / 2)), axis=0)
    for step in range(max_steps):
        quality = dprime(np.linalg.norm(layout - current, axis=1)) ** 2
        for index in range(n):
            variance = 1 / quality[index]
            if model == "C":
                observation = float(index == target) + latent[index] + rng.normal(0, np.sqrt(max(0, variance - 0.04)))
                residual_precision = 1 / max(1e-12, variance - 0.04)
                weighted_score[index] += residual_precision * observation
                independent_precision[index] += residual_precision
                effective_quality = 1 / (0.04 + 1 / independent_precision[index])
                llr[index] = effective_quality * (weighted_score[index] / independent_precision[index] - 0.5)
                best_quality[index] = effective_quality
            else:
                observation = float(index == target) + rng.normal(0, np.sqrt(variance))
                if model == "A":
                    llr[index] += quality[index] * (observation - 0.5)
                    best_quality[index] += quality[index]
                elif quality[index] > best_quality[index]:
                    llr[index] = quality[index] * (observation - 0.5)
                    best_quality[index] = quality[index]
        posterior = softmax(llr)
        if posterior.max() >= 0.95:
            return step + 1, int(posterior.argmax()) == target
        if strategy == "MAP":
            current = layout[int(posterior.argmax())]
        else:
            future = dprime(np.linalg.norm(omega[:, None, :] - layout[None, :, :], axis=2)) ** 2
            gains = ((future - best_quality[None, :]).clip(min=0) * posterior[None, :]).sum(axis=1)
            current = omega[int(gains.argmax())]
    return max_steps, int(posterior.argmax()) == target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("reports/spikes") / f"policy-toy-{stamp}.csv"
    with output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("layout", "observation_model", "strategy", "trials", "mean_steps", "accuracy", "max_step_fraction"))
        writer.writeheader()
        for layout_name, layout in LAYOUTS.items():
            for model in ("A", "B", "C"):
                for strategy in ("gain", "MAP"):
                    results = [one_episode(layout, trial % len(layout), model=model, strategy=strategy, seed=100000 + trial) for trial in range(args.trials)]
                    row = {"layout": layout_name, "observation_model": model, "strategy": strategy, "trials": args.trials, "mean_steps": float(np.mean([item[0] for item in results])), "accuracy": float(np.mean([item[1] for item in results])), "max_step_fraction": float(np.mean([item[0] == 15 for item in results]))}
                    writer.writerow(row)
                    print(json.dumps(row), flush=True)
    print(f"output={output}")


if __name__ == "__main__":
    main()
