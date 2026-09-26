import csv
import json

import pytest

from saccadenet.viz.figures import build_plot_data


def test_plot_data_requires_all_registered_jobs_and_counts_failures(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    cfg = {
        "methods": ["saccadenet_lite", "full_res_sliding"],
        "resolutions": [[1920, 1080]],
        "development_seed_start": 20050,
        "development_smoke_count": 2,
    }
    (run / "manifest.json").write_text(json.dumps({"split": "development"}), encoding="utf-8")
    (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with (run / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("method", "width", "height", "seed", "status", "target_hit", "semantic_flops", "sensing_flops", "sensing_bytes", "episode_seconds"))
        writer.writeheader()
        for method in cfg["methods"]:
            writer.writerow({"method": method, "width": 1920, "height": 1080, "seed": 20050, "status": "ok", "target_hit": 1, "semantic_flops": 100, "sensing_flops": 20, "sensing_bytes": 30, "episode_seconds": 0.1})
            writer.writerow({"method": method, "width": 1920, "height": 1080, "seed": 20051, "status": "error", "target_hit": 0, "semantic_flops": 0, "sensing_flops": 0, "sensing_bytes": 0, "episode_seconds": 0.1})
    rows = build_plot_data(run, bootstrap=10)
    assert len(rows) == 2
    assert all(row["n"] == 2 and row["accuracy"] == 0.5 and row["completed_n"] == 1 for row in rows)
    assert all(row["mean_semantic_flops"] == 50 for row in rows)
    with (run / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("method", "width", "height", "seed"))
        writer.writeheader()
    with pytest.raises(ValueError, match="incomplete"):
        build_plot_data(run)
