"""Cross-check E1 episodes -> summary -> plot data -> trace, and print Markdown evidence."""

import argparse
import collections
import csv
import json
import math
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def crosscheck(run_dir: Path, plot_dir: Path) -> tuple[list[str], bool]:
    episodes = _read_csv(run_dir / "episodes.csv")
    latest = {}
    for row in episodes:
        latest[(row["method"], int(row["width"]), int(row["seed"]))] = row
    summary = {(row["method"], int(row["width"])): row for row in _read_csv(run_dir / "summary.csv")}
    plot = {(row["method"], int(row["width"])): row for row in _read_csv(plot_dir / "plot-data.csv")}
    lines, ok = [], True

    groups = collections.defaultdict(list)
    for (method, width, _), row in latest.items():
        groups[method, width].append(row)
    checks = [("saccadenet_lite", 1920), ("saccadenet_lite", 15360), ("full_res_sliding", 15360), ("downsample_1stage", 3840), ("two_stage", 15360)]
    lines += ["| 方法 | 宽 | n | 命中(逐局/summary/图) | 语义FLOPs均值(逐局/summary/图) | 一致 |", "|---|---:|---:|---|---|---|"]
    for key in checks:
        rows = groups[key]
        n = len(rows)
        hits = sum(row["status"] == "ok" and row["target_hit"] == "1" for row in rows)
        semantic = sum(float(row["semantic_flops"] or 0) for row in rows) / n
        s, p = summary[key], plot[key]
        same = (n == int(s["planned_n"]) == int(p["n"]) and hits == int(s["hits"]) == int(p["hits"])
                and _close(semantic, float(s["mean_semantic_flops"])) and _close(semantic, float(p["mean_semantic_flops"])))
        ok &= same
        lines.append(f"| {key[0]} | {key[1]} | {n} | {hits}/{s['hits']}/{p['hits']} | {semantic:.6g}/{float(s['mean_semantic_flops']):.6g}/{float(p['mean_semantic_flops']):.6g} | {'是' if same else '否'} |")

    traces = {}
    for path in sorted(run_dir.glob("trace*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if "key" in item:
                    method, width, _, seed = item["key"]
                    traces[method, int(width), int(seed)] = item
    lines += ["", "| 逐局键 | steps(CSV/trace) | 语义FLOPs(CSV/trace末步) | 感知字节(CSV/trace) | 一致 |", "|---|---|---|---|---|"]
    for key in [("saccadenet_lite", 15360, seed) for seed in range(30000, 30005)]:
        row, item = latest[key], traces[key]
        last = item["steps"][-1]["costs"]
        same = (int(row["steps"]) == len(item["steps"]) and int(row["semantic_flops"]) == int(item["costs"]["semantic_flops"]) == int(last["semantic_flops"])
                and int(row["sensing_bytes"]) == int(item["costs"]["sensing_bytes"]))
        ok &= same
        lines.append(f"| {key[0]}:{key[1]}:{key[2]} | {row['steps']}/{len(item['steps'])} | {row['semantic_flops']}/{last['semantic_flops']} | {row['sensing_bytes']}/{item['costs']['sensing_bytes']} | {'是' if same else '否'} |")

    lines += ["", "SaccadeNet终止原因与未命中局：", ""]
    for width in (1920, 3840, 7680, 15360):
        rows = [latest[key] for key in latest if key[0] == "saccadenet_lite" and key[1] == width]
        reasons = collections.Counter(row["reason"] for row in rows)
        recall = sum(row["target_recall"] == "1" for row in rows)
        misses = [f"{row['seed']}({row['reason']},{row['steps']}眼)" for row in rows if row["target_hit"] != "1"]
        lines.append(f"- {width}：{dict(reasons)}；目标候选48px召回 {recall}/{len(rows)}；未命中 {', '.join(misses) or '无'}")
    return lines, ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    lines, ok = crosscheck(args.run, args.plot)
    print("\n".join(lines))
    print(f"\nALL_CONSISTENT={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
