import numpy as np
import torch


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


def test_best_observation_disregards_evidence_before_candidate_reset():
    from saccadenet.exp.recovery_diagnostics import best_observation
    steps = [
        {"reset_ids": [], "observations": [{"candidate_id": 4, "dprime": 3.0, "normalized_score": 2.0}]},
        {"reset_ids": [4], "observations": [{"candidate_id": 4, "dprime": 2.0, "normalized_score": 1.0}]},
    ]
    assert best_observation(steps, 4)["dprime"] == 2.0


def test_retinal_probe_replays_observation_score_from_saved_fixation():
    from saccadenet.exp.recovery_probes import replay_observation
    from saccadenet.models.fovea import FoveaNet
    from saccadenet.bayes.calibrate import score_query_logits
    from saccadenet.retina.pyramid import build_pyramid
    from saccadenet.retina.reconstruct import candidate_view
    from saccadenet.retina.sampler import sample_retina

    image = np.full((1080, 1920, 3), 126, dtype=np.uint8)
    image[444:636, 864:1056] = 220
    model = FoveaNet().eval()
    fixation = (960.0, 540.0)
    xy = (960.0, 540.0)
    retina = sample_retina(build_pyramid(image), fixation)
    patch, _ = candidate_view(retina, xy)
    tensor = torch.from_numpy(np.stack([patch])).permute(0, 3, 1, 2).float() / 255
    with torch.inference_mode():
        expected = float(score_query_logits(model(tensor), 3)[0])
    replayed = replay_observation(image, fixation, xy, 3, model, torch.device("cpu"))
    assert abs(replayed["raw_score"] - expected) < 1e-5
    assert replayed["valid_fraction"] == 1.0


def test_retinal_probe_uses_same_batch_layout_as_episode_inference():
    from saccadenet.exp.recovery_probes import replay_observation

    class RecordingNet(torch.nn.Module):
        def forward(self, batch):
            self.input_stride = tuple(batch.stride())
            return torch.zeros((len(batch), 11), device=batch.device)

    image = np.full((1080, 1920, 3), 126, dtype=np.uint8)
    model = RecordingNet()
    replay_observation(image, (960.0, 540.0), (960.0, 540.0), 3, model, torch.device("cpu"))
    assert model.input_stride == (96 * 96 * 3, 1, 96 * 3, 3)
