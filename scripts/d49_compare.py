"""D49: check decisions identical to D48 holdout and report timing change."""
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
TIMING = {"attempt_id", "episode_seconds", "canvas_seconds", "peak_rss_bytes", "peak_cuda_bytes"}


def rows(run):
    with (run / "episodes.csv").open(encoding="utf-8") as file:
        return {(r["method"], int(r["width"]), int(r["height"]), int(r["seed"])): r for r in csv.DictReader(file)}


def traces(run):
    out = {}
    for line in (run / "trace.jsonl").open(encoding="utf-8"):
        record = json.loads(line)
        record.pop("attempt_id", None)
        out[tuple(record["key"])] = json.dumps(record, sort_keys=True)
    return out


old, new = rows(old_dir), rows(new_dir)
assert old.keys() == new.keys(), (len(old), len(new))
mismatch = [k for k in old if {c: v for c, v in old[k].items() if c not in TIMING} != {c: v for c, v in new[k].items() if c not in TIMING}]
told, tnew = traces(old_dir), traces(new_dir)
trace_mismatch = [k for k in told if told[k] != tnew.get(k)]
print(f"episodes {len(old)}; decision-field mismatches {len(mismatch)}; trace mismatches {len(trace_mismatch)}")
for k in (mismatch + trace_mismatch)[:10]:
    print("  mismatch", k)

summary = {}
for (method, w, h, seed), r in new.items():
    summary.setdefault((w, method), {"old": [], "new": [], "steps": []})
    summary[(w, method)]["old"].append(float(old[(method, w, h, seed)]["episode_seconds"]))
    summary[(w, method)]["new"].append(float(r["episode_seconds"]))
    summary[(w, method)]["steps"].append(int(r["steps"]))
print("width,method,old_mean_s,new_mean_s,change,old_median_s,new_median_s,mean_steps,new_ms_per_glimpse")
for (w, method), v in sorted(summary.items()):
    o, n = mean(v["old"]), mean(v["new"])
    print(f"{w},{method},{o:.4f},{n:.4f},{n / o - 1:+.1%},{median(v['old']):.4f},{median(v['new']):.4f},{mean(v['steps']):.2f},{1000 * sum(v['new']) / sum(v['steps']):.2f}")
print("width,sacc_over_opt_old,sacc_over_opt_new")
for w in sorted({w for w, _ in summary}):
    s, t = summary[(w, "saccadenet_lite")], summary[(w, "two_stage_optimized")]
    print(f"{w},{mean(s['old']) / mean(t['old']):.2f},{mean(s['new']) / mean(t['new']):.2f}")
