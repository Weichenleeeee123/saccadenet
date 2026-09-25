"""Export an offline HTML replay player from saved E1 traces (no model inference is rerun)."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from saccadenet.config import EpisodeConfig
from saccadenet.contracts import RetinaOut
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import sample_retina
from saccadenet.viz.export_replay import _episode_spec, _load_trace

TEMPLATE = Path(__file__).with_name("demo_template")
STAGE_WIDTH = 1280


def retina_view(retina: RetinaOut, gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild pixels at canvas coordinates (gx, gy) using only one retina sample.

    Same interpolation rules as ``reconstruct.candidate_view`` but on an arbitrary grid,
    so the demo shows exactly the information the network received at that fixation.
    """
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    dx = gx - retina.fixation_xy[0]
    dy = gy - retina.fixation_xy[1]
    half = retina.fovea.shape[0] / 2
    sectors = retina.logpolar.shape[1]
    delta = 2 * math.pi / sectors
    radial = (np.log(np.maximum(np.hypot(dx, dy), half) / half) / delta).astype(np.float32)
    angular = (np.mod(np.arctan2(dy, dx), 2 * math.pi) / delta).astype(np.float32)
    angular = np.where(angular >= sectors, 0, angular).astype(np.float32)
    strip = np.concatenate((retina.logpolar, retina.logpolar[:, :1]), axis=1)
    strip_valid = np.concatenate((retina.logpolar_valid, retina.logpolar_valid[:, :1]), axis=1).astype(np.float32)
    outer = cv2.remap(strip, angular, radial, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    outer_valid = cv2.remap(strip_valid, angular, radial, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 0.999
    fx = (dx + retina.fovea.shape[1] / 2).astype(np.float32)
    fy = (dy + retina.fovea.shape[0] / 2).astype(np.float32)
    inner = cv2.remap(retina.fovea, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inner_valid = cv2.remap(retina.fovea_valid.astype(np.uint8), fx, fy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    in_fovea = (fx >= 0) & (fx < retina.fovea.shape[1]) & (fy >= 0) & (fy < retina.fovea.shape[0])
    view = np.where(in_fovea[..., None], inner, outer)
    valid = np.where(in_fovea, inner_valid, outer_valid)
    height, width = retina.canvas_shape
    valid &= (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    return view.astype(np.uint8), valid


def _grid(width: int, height: int, out_w: int, out_h: int, center=None, span=None):
    if center is None:
        xs = (np.arange(out_w, dtype=np.float32) + 0.5) * width / out_w - 0.5
        ys = (np.arange(out_h, dtype=np.float32) + 0.5) * height / out_h - 0.5
    else:
        step = span / out_w  # square display pixels
        xs = center[0] + (np.arange(out_w, dtype=np.float32) + 0.5 - out_w / 2) * step
        ys = center[1] + (np.arange(out_h, dtype=np.float32) + 0.5 - out_h / 2) * step
    return np.meshgrid(xs, ys)


def _save_jpeg(array: np.ndarray, valid: np.ndarray | None, path: Path) -> None:
    if valid is not None:
        array = np.where(valid[..., None], array, np.uint8(18)).astype(np.uint8)
    Image.fromarray(array).save(path, quality=82, optimize=True)


def _baselines(run_dir: Path, width: int, seed: int) -> dict:
    result = {}
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row["width"]) == width and int(row["seed"]) == seed and row["status"] == "ok":
                result[row["method"]] = {
                    "semantic_flops": float(row["semantic_flops"]), "sensing_bytes": float(row["sensing_bytes"]),
                    "seconds": float(row["episode_seconds"]), "hit": int(row["target_hit"]), "steps": int(row["steps"]),
                }
    return result


def export_episode(run_dir: Path, spec: str, target: Path, *, note: str) -> dict:
    key = _episode_spec(spec)
    if key[0] != "saccadenet_lite":
        raise ValueError("demo requires a SaccadeNet trace")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    trace = _load_trace(run_dir, key)
    config = EpisodeConfig(width=key[1], height=key[2], query=trace["query"])
    image, truth = make_canvas(config, key[3], MnistBank.from_split(manifest["split"]))
    target_xy = tuple(map(float, truth.centers_xy[truth.target_index]))
    if truth.target_index != trace["truth_target_index"] or not np.allclose(target_xy, trace["truth_target_xy"]):
        raise ValueError("regenerated canvas does not match saved evaluation truth")
    episode_id = f"{key[1]}x{key[2]}-{key[3]}"
    folder = target / episode_id
    folder.mkdir(parents=True)
    stage_h = round(STAGE_WIDTH * key[2] / key[1])
    Image.fromarray(cv2.resize(image, (STAGE_WIDTH, stage_h), interpolation=cv2.INTER_AREA)).save(folder / "canvas.jpg", quality=85, optimize=True)
    pyramid = build_pyramid(image)
    zoom_span = min(key[1], 2400)
    steps = []
    for index, step in enumerate(trace["steps"]):
        xy = tuple(step["fixation_xy"])
        retina = sample_retina(pyramid, xy, fovea_size=config.fovea_size, sectors=config.sectors)
        seen, valid = retina_view(retina, *_grid(key[1], key[2], STAGE_WIDTH, stage_h))
        _save_jpeg(seen, valid, folder / f"seen-{index + 1:03d}.jpg")
        zoom, zoom_valid = retina_view(retina, *_grid(key[1], key[2], 480, 270, center=xy, span=zoom_span))
        _save_jpeg(zoom, zoom_valid, folder / f"zoom-{index + 1:03d}.jpg")
        Image.fromarray(retina.fovea).resize((192, 192), Image.Resampling.NEAREST).save(folder / f"fovea-{index + 1:03d}.png")
        steps.append({
            "fixation": [float(v) for v in xy],
            "candidates": [{"id": int(c["id"]), "xy": [float(v) for v in c["xy"]]} for c in step["candidates"]],
            "posterior": [float(p) for p in step["posterior"]],
            "semantic_flops": float(step["costs"]["semantic_flops"]),
            "sensing_flops": float(step["costs"]["sensing_flops"]),
            "sensing_bytes": float(step["costs"]["sensing_bytes"]),
        })
    baselines = _baselines(run_dir, key[1], key[3])
    own = baselines.get("saccadenet_lite")
    if own is None or own["steps"] != len(steps) or not math.isclose(own["semantic_flops"], steps[-1]["semantic_flops"]):
        raise ValueError(f"demo frames disagree with episodes.csv for {spec}")
    return {
        "id": episode_id, "spec": spec, "note": note, "width": key[1], "height": key[2], "seed": key[3],
        "query": int(trace["query"]), "reason": trace["reason"], "attempt_id": trace["attempt_id"],
        "answer": [float(v) for v in trace["answer_xy"]] if trace["answer_xy"] is not None else None,
        "truth": [float(v) for v in target_xy], "zoom_span": zoom_span,
        "stage": [STAGE_WIDTH, stage_h], "steps": steps, "baselines": baselines,
    }


def export_demo(run_dir: Path, episodes: list[tuple[str, str]], out: Path) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"demo-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    for name in ("index.html", "player.js", "style.css"):
        shutil.copyfile(TEMPLATE / name, target / name)
    data = {
        "source_run": run_dir.name, "git_head": manifest["git_head"], "checkpoint_sha256": manifest["checkpoint_sha256"],
        "generated_utc": stamp, "note": "Frames regenerated deterministically from recorded seeds and saved traces; no inference rerun.",
        "episodes": [export_episode(run_dir, spec, target, note=note) for spec, note in episodes],
    }
    (target / "data.js").write_text("window.DEMO_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    manifest_dir = Path("reports/replay")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "demo_dir": str(target), "source_run": run_dir.name, "source_git_head": manifest["git_head"],
        "checkpoint_sha256": manifest["checkpoint_sha256"], "generated_utc": stamp,
        "checks": "frame count == trace steps == episodes.csv steps; last-frame semantic FLOPs == episodes.csv; regenerated target == saved truth",
        "episodes": [{"spec": e["spec"], "selection_reason": e["note"], "frames": len(e["steps"]), "reason": e["reason"],
                      "hit": e["baselines"]["saccadenet_lite"]["hit"]} for e in data["episodes"]],
        "open": f"python -m http.server 8765 --directory {out}  then  http://127.0.0.1:8765/{target.name}/index.html",
    }
    with (manifest_dir / f"manifest-{stamp}.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", action="append", required=True, help="method:WIDTHxHEIGHT:seed=说明")
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()
    episodes = [tuple(item.split("=", 1)) if "=" in item else (item, "") for item in args.episode]
    print(export_demo(args.run, episodes, args.out))


if __name__ == "__main__":
    main()
