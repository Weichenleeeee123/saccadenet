import json

import numpy as np
import pytest

from saccadenet.exp.store import ExperimentStore, episode_key


MANIFEST = {
    "config_sha256": "a", "checkpoint_sha256": "b", "retina_fit_sha256": "c",
    "two_stage_fit_sha256": "d", "split": "development",
}


def test_resume_skips_success_and_preserves_failed_attempt(tmp_path):
    run_dir = tmp_path / "run"
    store = ExperimentStore(run_dir, MANIFEST, create=True)
    first = episode_key("method", 1920, 1080, 1)
    failed = episode_key("method", 1920, 1080, 2)
    store.record({"attempt_id": 1, "method": first[0], "width": first[1], "height": first[2], "seed": first[3], "status": "ok"}, trace={"steps": [], "target": np.int32(3)})
    assert json.loads((run_dir / "trace.jsonl").read_text(encoding="utf-8"))["target"] == 3
    store.record({"attempt_id": 1, "method": failed[0], "width": failed[1], "height": failed[2], "seed": failed[3], "status": "error"}, error={"type": "test"})
    resumed = ExperimentStore(run_dir, MANIFEST, create=False)
    assert not resumed.needs_run(first)
    assert resumed.needs_run(failed)
    assert resumed.next_attempt(failed) == 2
    with pytest.raises(ValueError, match="already recorded"):
        resumed.record({"attempt_id": 2, "method": first[0], "width": first[1], "height": first[2], "seed": first[3], "status": "ok"})
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        ExperimentStore(run_dir, {**MANIFEST, "checkpoint_sha256": "changed"}, create=False)


def test_truncated_trace_is_retained_and_new_segment_is_used(tmp_path):
    run_dir = tmp_path / "run"
    ExperimentStore(run_dir, MANIFEST, create=True)
    with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"incomplete":')
    resumed = ExperimentStore(run_dir, MANIFEST, create=False)
    assert resumed.trace_path.name == "trace-resume-1.jsonl"
    assert (run_dir / "trace.jsonl").read_text(encoding="utf-8") == '{"incomplete":'
    assert "truncated" in resumed.trace_path.read_text(encoding="utf-8")
def test_balanced_method_order_rotates_each_seed():
    from saccadenet.exp import e1_resolution
    assert hasattr(e1_resolution, "balanced_method_order")
    methods = ["saccadenet_lite", "two_stage", "two_stage_no_coarse"]
    assert e1_resolution.balanced_method_order(methods, 0) == methods
    assert e1_resolution.balanced_method_order(methods, 1) == methods[1:] + methods[:1]
    assert e1_resolution.balanced_method_order(methods, 2) == methods[2:] + methods[:2]
