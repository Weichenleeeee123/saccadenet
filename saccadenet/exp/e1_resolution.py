"""E1 four-resolution comparison with append-only resume and truth-only scoring."""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import psutil
import torch
import yaml

from saccadenet.bayes.calibrate import GaussianCalibrator
from saccadenet.config import EpisodeConfig
from saccadenet.contracts import EpisodeInput
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.exp.store import ExperimentStore, episode_key
from saccadenet.models.fovea import FoveaNet
from saccadenet.retina.horizon import derived_grid, detection_horizon
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.run.baselines import PlattCalibration, ideal_dense_flops, run_downsample_1stage, run_full_res_sliding, run_two_stage
from saccadenet.run.episode import SceneSensor, run_episode
from saccadenet.run.evaluate import evaluate_answer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(cfg: dict, split: str, *, config_path: Path) -> dict:
    checkpoint, retina_fit, two_stage_fit = (Path(cfg[key]) for key in ("checkpoint", "retina_fit", "two_stage_fit"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    return {
        "name": cfg["name"], "split": split, "config_path": str(config_path),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "retina_fit": str(retina_fit), "retina_fit_sha256": _sha256(retina_fit),
        "two_stage_fit": str(two_stage_fit), "two_stage_fit_sha256": _sha256(two_stage_fit),
        "git_head": head, "git_dirty": dirty, "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def planned_jobs(cfg: dict, *, smoke: bool = False) -> list[tuple[str, int, int, int]]:
    start = cfg["development_seed_start"] if smoke else cfg["test_seed_start"]
    count = cfg["development_smoke_count"] if smoke else cfg["test_seed_count"]
    result = []
    for width, height in cfg["resolutions"]:
        for seed in range(start, start + count):
            for method in cfg["methods"]:
                if not smoke and method == "full_res_sliding" and width == 15360 and seed >= start + cfg["full_16k_count"]:
                    continue
                result.append((method, width, height, seed))
    return result


def balanced_method_order(methods: list[str], seed: int) -> list[str]:
    """Rotate methods across successive seeds to balance execution order."""
    if not methods:
        return []
    offset = seed % len(methods)
    return methods[offset:] + methods[:offset]


def _wilson(hits: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = hits / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return center - radius, center + radius


def write_summary(run_dir: Path, jobs: list[tuple[str, int, int, int]]) -> Path:
    latest: dict[tuple[str, int, int, int], dict] = {}
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = episode_key(row["method"], row["width"], row["height"], row["seed"])
            latest[key] = row
    groups: dict[tuple[str, int, int], list[dict | None]] = {}
    for method, width, height, seed in jobs:
        groups.setdefault((method, width, height), []).append(latest.get((method, width, height, seed)))
    output = run_dir / "summary.csv"
    if output.exists():
        # Preserve previous summaries for audit, then create a new numbered snapshot.
        index = 2
        while (run_dir / f"summary-{index}.csv").exists():
            index += 1
        output = run_dir / f"summary-{index}.csv"
    with output.open("x", newline="", encoding="utf-8") as file:
        fields = ("method", "width", "height", "planned_n", "recorded_n", "completed_n", "hits", "accuracy", "wilson_low", "wilson_high", "mean_semantic_flops", "ideal_semantic_flops", "mean_sensing_flops", "mean_sensing_bytes", "mean_episode_seconds")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (method, width, height), rows in groups.items():
            recorded = [row for row in rows if row is not None]
            completed = [row for row in recorded if row["status"] == "ok"]
            hits = sum(int(row["target_hit"]) for row in completed)
            low, high = _wilson(hits, len(rows))
            def mean(field: str) -> float:
                return float(np.mean([float(row[field] or 0) for row in recorded])) if recorded else float("nan")
            writer.writerow({"method": method, "width": width, "height": height, "planned_n": len(rows), "recorded_n": len(recorded), "completed_n": len(completed), "hits": hits, "accuracy": hits / len(rows), "wilson_low": low, "wilson_high": high, "mean_semantic_flops": mean("semantic_flops"), "ideal_semantic_flops": ideal_dense_flops(FoveaNet(), width, height) if method == "full_res_sliding" else "", "mean_sensing_flops": mean("sensing_flops"), "mean_sensing_bytes": mean("sensing_bytes"), "mean_episode_seconds": mean("episode_seconds")})
    return output


def run_job(
    method: str, image: np.ndarray, config: EpisodeConfig, network: FoveaNet,
    retina_calibrator: GaussianCalibrator, two_stage_calibrator: PlattCalibration,
    *, device: torch.device, tile_outputs: int, two_stage_threshold: float,
):
    if method in ("saccadenet_lite", "saccadenet_derived_grid", "saccadenet_confirm_unscored"):
        if method == "saccadenet_derived_grid":
            # D27: grid from the analytic horizon, keeping the frozen 15-glimpse verification margin (40 - 5*5).
            grid = derived_grid(config.width, config.height, detection_horizon(config.card_size, config.sectors)[0])
            config = replace(config, exploration_grid=grid, t_max=grid * grid + 15)
        sensor = SceneSensor(build_pyramid(image), fovea_size=config.fovea_size, sectors=config.sectors)
        return run_episode(
            sensor, EpisodeInput(config.query, config.width, config.height, config.k),
            config, retina_calibrator, network, device=device,
            require_all_scored=method == "saccadenet_confirm_unscored",
        )
    if method == "full_res_sliding":
        return run_full_res_sliding(image, config.query, network, device=device, tile_outputs=tile_outputs)
    if method == "downsample_1stage":
        return run_downsample_1stage(image, config.query, network, device=device, tile_outputs=tile_outputs)
    if method in ("two_stage", "two_stage_no_coarse"):
        return run_two_stage(image, config.query, network, device=device, threshold=two_stage_threshold, probability_calibration=two_stage_calibrator, coarse_ranking=method == "two_stage")
    raise ValueError(f"unknown method {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/e1.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    jobs = planned_jobs(cfg, smoke=args.smoke)
    split = "development" if args.smoke else "final_test"
    manifest = _manifest(cfg, split, config_path=args.config)
    if args.dry_run:
        print(json.dumps({"split": split, "jobs": len(jobs), "methods": cfg["methods"], "resolutions": cfg["resolutions"], "checkpoint_sha256": manifest["checkpoint_sha256"], "retina_fit_sha256": manifest["retina_fit_sha256"], "two_stage_fit_sha256": manifest["two_stage_fit_sha256"], "git_head": manifest["git_head"], "git_dirty": manifest["git_dirty"]}, indent=2))
        return
    if args.resume is not None:
        store = ExperimentStore(args.resume, manifest, create=False)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        tag = cfg.get("run_tag", "e1")
        run_dir = Path("runs") / f"{stamp}-{manifest['git_head'][:8]}-{tag}-{'smoke' if args.smoke else 'final'}"
        store = ExperimentStore(run_dir, manifest, create=True)
        with (store.run_dir / "config.json").open("x", encoding="utf-8") as file:
            json.dump(cfg, file, indent=2)
        with (store.run_dir / "seeds.csv").open("x", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=("split", "width", "height", "seed", "digit_ids")).writeheader()
    print(f"run_dir={store.run_dir} remaining={sum(store.needs_run(key) for key in jobs)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    network = FoveaNet().to(device).eval()
    network.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)["model"])
    fit = json.loads(Path(cfg["retina_fit"]).read_text(encoding="utf-8"))
    retina_calibrator = GaussianCalibrator(**fit["model"])
    platt = json.loads(Path(cfg["two_stage_fit"]).read_text(encoding="utf-8"))
    two_stage_calibrator = PlattCalibration(platt["slope"], platt["intercept"])
    if int(cfg.get("warmup_forward_count", 0)):
        with torch.inference_mode():
            sample = torch.zeros((1, 3, 96, 96), device=device)
            for _ in range(int(cfg["warmup_forward_count"])):
                network(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    bank = MnistBank.from_split(split)
    recorded_seeds = set()
    with (store.run_dir / "seeds.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            recorded_seeds.add((int(row["width"]), int(row["height"]), int(row["seed"])))
    process = psutil.Process()
    grouped: dict[tuple[int, int, int], list[str]] = {}
    for method, width, height, seed in jobs:
        if store.needs_run((method, width, height, seed)):
            grouped.setdefault((width, height, seed), []).append(method)
    for (width, height, seed), methods in grouped.items():
        # K defaults to the frozen E1 value; E2 (D29) varies it per run.
        config = EpisodeConfig(width=width, height=height, query=seed % 10, k=int(cfg.get("k", 12)))
        canvas_start = time.perf_counter()
        try:
            image, truth = make_canvas(config, seed, bank)
            canvas_seconds = time.perf_counter() - canvas_start
            if (width, height, seed) not in recorded_seeds:
                with (store.run_dir / "seeds.csv").open("a", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=("split", "width", "height", "seed", "digit_ids")).writerow({"split": split, "width": width, "height": height, "seed": seed, "digit_ids": json.dumps(truth.digit_ids)})
                recorded_seeds.add((width, height, seed))
        except Exception as exc:
            image = truth = None
            canvas_seconds = time.perf_counter() - canvas_start
            canvas_error = exc
        for method in (balanced_method_order(methods, seed) if cfg.get("balanced_method_order", False) else methods):
            key = (method, width, height, seed)
            attempt_id = store.next_attempt(key)
            row = {"attempt_id": attempt_id, "method": method, "width": width, "height": height, "seed": seed, "canvas_seconds": canvas_seconds}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                if image is None:
                    raise canvas_error
                result = run_job(method, image, config, network, retina_calibrator, two_stage_calibrator, device=device, tile_outputs=cfg["tile_outputs"], two_stage_threshold=cfg["two_stage_threshold"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                evaluation = evaluate_answer(result.answer_xy, truth, match_radius=cfg["match_radius"])
                candidates = result.trace[-1].get("candidates", []) if result.trace else []
                target_xy = tuple(truth.centers_xy[truth.target_index])
                target_recall = int(any(math.dist(item["xy"], target_xy) <= cfg["match_radius"] for item in candidates)) if method.startswith("saccadenet") else ""
                row.update({"status": "ok", "target_hit": int(evaluation.target_hit), "reason": result.reason, "steps": result.steps, "candidate_count": len(candidates) if method.startswith("saccadenet") else "", "target_recall": target_recall, "matched_card_index": evaluation.matched_card_index, "localization_error": evaluation.localization_error, "sensing_flops": result.costs.get("sensing_flops", 0), "sensing_bytes": result.costs.get("sensing_bytes", 0), "semantic_flops": result.costs.get("semantic_flops", 0), "episode_seconds": time.perf_counter() - started, "peak_rss_bytes": process.memory_info().rss, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "answer_x": result.answer_xy[0] if result.answer_xy else "", "answer_y": result.answer_xy[1] if result.answer_xy else ""})
                trace = {"key": key, "attempt_id": attempt_id, "answer_xy": result.answer_xy, "truth_target_xy": target_xy, "truth_target_index": truth.target_index, "query": config.query, "reason": result.reason, "steps": result.trace, "costs": result.costs}
                store.record(row, trace=trace)
            except Exception as exc:
                row.update({"status": "error", "target_hit": 0, "reason": "exception", "episode_seconds": time.perf_counter() - started, "peak_rss_bytes": process.memory_info().rss, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "error_type": type(exc).__name__})
                store.record(row, error={"key": key, "attempt_id": attempt_id, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            print(json.dumps({"key": key, "status": row["status"], "hit": row.get("target_hit", 0), "reason": row["reason"], "seconds": row["episode_seconds"]}), flush=True)
    summary = write_summary(store.run_dir, jobs)
    print(f"summary={summary} completed={len(store.completed)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
