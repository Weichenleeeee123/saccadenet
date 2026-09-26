import numpy as np
import pytest
import torch

from saccadenet.bayes.calibrate import fit_gaussian, score_query_logits, choose_inbounds_fixation


def test_gaussian_fit_recovers_signal_and_normalized_means():
    rng = np.random.default_rng(4)
    negatives = rng.normal(0, 1, 5000)
    positives = rng.normal(2, 1, 5000)
    scores = np.concatenate((negatives, positives))
    labels = np.concatenate((np.zeros(5000), np.ones(5000)))
    calibrator = fit_gaussian(scores, labels, np.full(10000, 4.0), edges=(0, 10))
    assert abs(calibrator.dprime(4) - 2.0) < 0.1
    assert abs(np.mean([calibrator.normalize(v, 4) for v in negatives])) < 0.06
    assert abs(np.mean([calibrator.normalize(v, 4) for v in positives]) - 1) < 0.06


def test_low_signal_bin_is_skipped_without_nonfinite_evidence():
    rng = np.random.default_rng(12)
    negatives = rng.normal(0, 1, 2000)
    positives = rng.normal(0.01, 1, 2000)
    calibrator = fit_gaussian(
        np.r_[negatives, positives],
        np.r_[np.zeros(2000), np.ones(2000)],
        np.full(4000, 2.0),
        edges=(0, 10),
    )
    assert calibrator.dprime(2) < 0.1
    assert calibrator.normalize(0.5, 2) is None


def test_query_log_odds_uses_all_other_classes():
    logits = torch.tensor([[4.0, 1.0, 0.0]])
    score = score_query_logits(logits, query=0)
    expected = 4.0 - torch.logsumexp(logits[:, 1:], dim=1)
    assert torch.allclose(score, expected)


def test_missing_class_in_bin_is_reported():
    with pytest.raises(ValueError, match="class"):
        fit_gaussian(np.ones(10), np.ones(10), np.ones(10), edges=(0, 10))


@pytest.mark.parametrize("center", [(960, 540), (96, 96), (1824, 984)])
def test_calibration_fixation_stays_inside_canvas_at_requested_distance(center):
    chosen = choose_inbounds_fixation(center, eccentricity=1050, canvas_shape=(1080, 1920))
    assert chosen is not None
    assert 0 <= chosen[0] < 1920 and 0 <= chosen[1] < 1080
    assert abs(np.linalg.norm(np.asarray(chosen) - center) - 1050) < 1e-3
