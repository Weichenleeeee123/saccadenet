"""Append-only experiment store with hash-checked resume."""

import csv
import json
from pathlib import Path

import numpy as np


EPISODE_FIELDS = (
    "attempt_id", "method", "width", "height", "seed", "status", "target_hit", "reason",
    "steps", "candidate_count", "target_recall", "matched_card_index", "localization_error",
    "sensing_flops", "sensing_bytes", "semantic_flops", "episode_seconds", "canvas_seconds", "peak_rss_bytes",
    "peak_cuda_bytes", "answer_x", "answer_y", "error_type",
)


def episode_key(method: str, width: int, height: int, seed: int) -> tuple[str, int, int, int]:
    return method, int(width), int(height), int(seed)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


class ExperimentStore:
    def __init__(self, run_dir: Path, manifest: dict, *, create: bool) -> None:
        self.run_dir = Path(run_dir)
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            with (self.run_dir / "manifest.json").open("x", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)
            with (self.run_dir / "episodes.csv").open("x", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=EPISODE_FIELDS).writeheader()
            (self.run_dir / "trace.jsonl").open("x", encoding="utf-8").close()
            (self.run_dir / "errors.jsonl").open("x", encoding="utf-8").close()
        else:
            existing = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
            for field in ("config_sha256", "checkpoint_sha256", "retina_fit_sha256", "two_stage_fit_sha256", "split"):
                if existing[field] != manifest[field]:
                    raise ValueError(f"resume rejected: {field} differs")
        self.completed: set[tuple[str, int, int, int]] = set()
        self.attempts: dict[tuple[str, int, int, int], int] = {}
        with (self.run_dir / "episodes.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
                raise ValueError("resume rejected: episodes.csv schema differs")
            for row in reader:
                key = episode_key(row["method"], row["width"], row["height"], row["seed"])
                self.attempts[key] = max(self.attempts.get(key, 0), int(row["attempt_id"]))
                if row["status"] == "ok":
                    self.completed.add(key)
        self.trace_path = self._appendable_jsonl("trace")
        self.errors_path = self._appendable_jsonl("errors")

    def _appendable_jsonl(self, stem: str) -> Path:
        base = self.run_dir / f"{stem}.jsonl"
        if not base.exists() or base.stat().st_size == 0:
            return base
        with base.open("rb") as file:
            file.seek(-1, 2)
            valid_end = file.read() == b"\n"
        if valid_end:
            return base
        index = 1
        while (self.run_dir / f"{stem}-resume-{index}.jsonl").exists():
            index += 1
        replacement = self.run_dir / f"{stem}-resume-{index}.jsonl"
        with replacement.open("x", encoding="utf-8") as file:
            file.write(json.dumps({"warning": f"{base.name} has a truncated final line; retained unchanged"}) + "\n")
        return replacement

    def needs_run(self, key: tuple[str, int, int, int]) -> bool:
        return key not in self.completed

    def next_attempt(self, key: tuple[str, int, int, int]) -> int:
        return self.attempts.get(key, 0) + 1

    def record(self, row: dict, *, trace: dict | None = None, error: dict | None = None) -> None:
        key = episode_key(row["method"], row["width"], row["height"], row["seed"])
        if key in self.completed:
            raise ValueError("successful episode already recorded")
        expected_attempt = self.next_attempt(key)
        if int(row["attempt_id"]) != expected_attempt:
            raise ValueError("attempt_id must increase without overwriting history")
        if trace is not None:
            with self.trace_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(trace, default=_json_default) + "\n")
                file.flush()
        if error is not None:
            with self.errors_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(error, default=_json_default) + "\n")
                file.flush()
        with (self.run_dir / "episodes.csv").open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
            writer.writerow({field: row.get(field, "") for field in EPISODE_FIELDS})
            file.flush()
        self.attempts[key] = expected_attempt
        if row["status"] == "ok":
            self.completed.add(key)
