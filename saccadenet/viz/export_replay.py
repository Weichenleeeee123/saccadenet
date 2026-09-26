"""Export numbered offline PNG frames from a saved Level 2 trace."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import sample_retina


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _episode_spec(spec: str) -> tuple[str, int, int, int]:
    method, resolution, seed = spec.split(":")
    width, height = map(int, resolution.split("x"))
    return method, width, height, int(seed)


def _load_trace(run_dir: Path, key: tuple[str, int, int, int]) -> dict:
    chosen = None
    for path in sorted(run_dir.glob("trace*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # The original truncated segment remains untouched.
                if tuple(item.get("key", ())) == key and (chosen is None or item["attempt_id"] > chosen["attempt_id"]):
                    chosen = item
    if chosen is None:
        raise ValueError(f"trace not found for {key}")
    return chosen


def export_replay(run_dir: Path, episode: str, *, out: Path = Path("artifacts/replay")) -> Path:
    run_dir = Path(run_dir)
    key = _episode_spec(episode)
    if key[0] != "saccadenet_lite":
        raise ValueError("replay requires a SaccadeNet Level 2 trace")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    trace = _load_trace(run_dir, key)
    if not trace["steps"]:
        raise ValueError("episode has no observation frames")
    config = EpisodeConfig(width=key[1], height=key[2], query=trace["query"])
    image, truth = make_canvas(config, key[3], MnistBank.from_split(manifest["split"]))
    target_xy = tuple(map(float, truth.centers_xy[truth.target_index]))
    if truth.target_index != trace["truth_target_index"] or not np.allclose(target_xy, trace["truth_target_xy"]):
        raise ValueError("regenerated canvas does not match saved evaluation truth")
    pyramid = build_pyramid(image)
    thumbnail = Image.fromarray(image).resize((800, 450), Image.Resampling.BILINEAR)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"{run_dir.name}-{key[1]}x{key[2]}-{key[3]}-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    seen_fixations: list[tuple[float, float]] = []
    font = _font(19)
    small_font = _font(15)
    scale_x, scale_y = 800 / key[1], 450 / key[2]
    for step_index, step in enumerate(trace["steps"]):
        frame = Image.new("RGB", (1240, 690), (245, 247, 249))
        painter = ImageDraw.Draw(frame)
        frame.paste(thumbnail, (20, 55))
        xy = tuple(step["fixation_xy"])
        seen_fixations.append(xy)
        def mapped(point):
            return (20 + point[0] * scale_x, 55 + point[1] * scale_y)
        path = [mapped(point) for point in seen_fixations]
        if len(path) > 1:
            painter.line(path, fill=(0, 110, 119), width=3)
        for index, point in enumerate(path):
            x, y = point
            painter.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(0, 110, 119), outline="white", width=2)
            painter.text((x + 7, y - 10), str(index + 1), fill="white", font=small_font, stroke_width=2, stroke_fill=(0, 75, 80))
        ranked = sorted(zip(step["candidates"], step["posterior"]), key=lambda pair: -pair[1])
        labels_to_show = {item["id"] for item, _ in ranked[:3]}
        for item, probability in zip(step["candidates"], step["posterior"]):
            x, y = mapped(item["xy"])
            painter.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(237, 92, 58), width=2)
            if item["id"] in labels_to_show:
                painter.text((x + 9, y - 8), f"{item['id']}:{probability:.2f}", fill=(74, 30, 19), font=small_font, stroke_width=2, stroke_fill="white")
        if step_index == len(trace["steps"]) - 1:
            tx, ty = mapped(target_xy)
            painter.ellipse((tx - 12, ty - 12, tx + 12, ty + 12), outline=(23, 145, 80), width=4)
            if trace["answer_xy"] is not None:
                ax, ay = mapped(trace["answer_xy"])
                painter.line((ax - 10, ay - 10, ax + 10, ay + 10), fill=(30, 70, 185), width=4)
                painter.line((ax - 10, ay + 10, ax + 10, ay - 10), fill=(30, 70, 185), width=4)
        retina = sample_retina(pyramid, xy, fovea_size=config.fovea_size, sectors=config.sectors)
        fovea = Image.fromarray(retina.fovea).resize((192, 192), Image.Resampling.NEAREST)
        polar = Image.fromarray(retina.logpolar).resize((340, 180), Image.Resampling.NEAREST)
        frame.paste(fovea, (860, 80))
        frame.paste(polar, (860, 340))
        painter.text((20, 15), f"SaccadeNet | query {config.query} | seed {key[3]} | fixation {step_index + 1}/{len(trace['steps'])}", fill=(20, 32, 40), font=font)
        painter.text((860, 55), "Foveal input (96x96)", fill=(20, 32, 40), font=small_font)
        painter.text((860, 315), "Log-polar observation", fill=(20, 32, 40), font=small_font)
        costs = step["costs"]
        painter.text((20, 535), f"Candidates: {len(step['candidates'])}    Max posterior: {max(step['posterior'], default=0):.3f}", fill=(20, 32, 40), font=font)
        painter.text((20, 567), f"Semantic: {costs['semantic_flops'] / 1e6:.1f} M FLOPs    Sensing: {costs['sensing_flops'] / 1e6:.1f} M FLOPs", fill=(20, 32, 40), font=font)
        if step_index == len(trace["steps"]) - 1:
            painter.text((20, 600), f"Final: {trace['reason']} | green=true target (evaluation only), blue=answer", fill=(20, 32, 40), font=small_font)
        else:
            painter.text((20, 600), "Green target overlay appears only after the final decision.", fill=(20, 32, 40), font=small_font)
        frame.save(target / f"frame-{step_index + 1:03d}.png")
    with (target / "replay.json").open("x", encoding="utf-8") as file:
        json.dump({"source_run": str(run_dir), "episode_key": key, "attempt_id": trace["attempt_id"], "frame_count": len(trace["steps"]), "reason": trace["reason"], "git_head": manifest["git_head"], "regeneration": "deterministic canvas from recorded seed and MNIST split; no model inference rerun"}, file, indent=2)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", required=True, help="method:WIDTHxHEIGHT:seed")
    parser.add_argument("--format", choices=("png",), default="png")
    parser.add_argument("--out", type=Path, default=Path("artifacts/replay"))
    args = parser.parse_args()
    print(export_replay(args.run, args.episode, out=args.out))


if __name__ == "__main__":
    main()
