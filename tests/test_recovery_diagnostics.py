import numpy as np


def test_trace_diagnosis_separates_recall_scoring_and_confident_wrong_choice():
    from saccadenet.exp.recovery_diagnostics import summarize_trace

    centers = np.array([[100.0, 100.0], [500.0, 100.0]])
    trace = {
        "key": ["saccadenet_lite", 1920, 1080, 20110],
        "truth_target_index": 0,
        "reason": "threshold",
        "steps": [{
            "step": 0,
            "candidates": [{"id": 4, "xy": [110.0, 100.0]}, {"id": 8, "xy": [500.0, 100.0]}],
            "posterior": [0.03, 0.97],
            "reset_ids": [],
            "observations": [
                {"candidate_id": 4, "candidate_xy": [110.0, 100.0], "eccentricity": 10.0, "dprime": 2.0, "raw_score": -0.5, "normalized_score": 0.2, "valid_fraction": 1.0},
                {"candidate_id": 8, "candidate_xy": [500.0, 100.0], "eccentricity": 0.0, "dprime": 2.0, "raw_score": 2.0, "normalized_score": 1.8, "valid_fraction": 1.0},
            ],
        }],
    }
    row = summarize_trace(trace, centers)
    assert row["category"] == "confident_wrong"
    assert row["target_recalled"] is True
    assert row["target_scored"] is True
    assert row["target_offset"] == 10.0
    assert row["winner_offset"] == 0.0
    assert row["target_best_raw_score"] == -0.5
    assert row["winner_best_raw_score"] == 2.0
