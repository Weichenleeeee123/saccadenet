import numpy as np
import torch

from saccadenet.models.fovea import FoveaNet
from saccadenet.cost.accounting import count_model_flops
from saccadenet.run.baselines import _DenseAdapter, _crop, _dense_query_map_tiled, _coarse_card_centers, _window_starts, fit_platt, ideal_dense_flops, run_two_stage


def test_window_starts_cover_final_edge_without_duplicates():
    assert _window_starts(129) == [0, 16, 32, 33]
    assert _window_starts(96) == [0]


def test_tiled_dense_scores_match_whole_image_on_all_windows():
    torch.manual_seed(9)
    model = FoveaNet().eval()
    image = np.random.default_rng(9).integers(0, 256, (129, 161, 3), dtype=np.uint8)
    scores, _ = _dense_query_map_tiled(image, model, 3, device=torch.device("cpu"), tile_outputs=2)
    assert len(scores) == len(_window_starts(129)) * len(_window_starts(161))
    for (x, y), score in scores.items():
        patch = torch.from_numpy(image[y : y + 96, x : x + 96].copy()).permute(2, 0, 1)[None].float() / 255
        with torch.inference_mode():
            logits = model(patch)
            expected = float(logits[0, 3] - torch.logsumexp(torch.cat((logits[0, :3], logits[0, 4:])), dim=0))
        assert abs(score - expected) < 1e-4


def test_coarse_detector_handles_blank_and_two_cards():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    assert _coarse_card_centers(image, threshold=195) == []
    image[20:35, 25:40] = 225
    image[70:90, 80:100] = 225
    centers = _coarse_card_centers(image, threshold=195)
    assert len(centers) == 2
    assert any(abs(x - 32) < 2 and abs(y - 27) < 2 for x, y in centers)


def test_platt_calibration_orders_and_bounds_probabilities():
    scores = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    fit = fit_platt(scores, labels)
    probabilities = [fit.probability(score) for score in scores]
    assert all(0 < value < 1 for value in probabilities)
    assert probabilities == sorted(probabilities)
    assert probabilities[0] < 0.1 < 0.9 < probabilities[-1]


def test_ideal_dense_flops_matches_profile_and_tiled_cost_is_no_lower():
    model = FoveaNet().eval()
    whole = count_model_flops(_DenseAdapter(model), (1, 3, 129, 161))
    assert ideal_dense_flops(model, 161, 129) == whole
    image = np.zeros((129, 161, 3), dtype=np.uint8)
    _, measured = _dense_query_map_tiled(image, model, 0, device=torch.device("cpu"), tile_outputs=2, collect=False)
    assert measured["semantic_flops"] >= whole


def test_two_stage_empty_and_edge_crop_do_not_read_outside_image():
    image = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    result = run_two_stage(image, 3, FoveaNet().eval(), device=torch.device("cpu"))
    assert result.answer_xy is None
    assert result.reason == "no_candidates"
    assert _crop(image, (0, 0)).shape == (96, 96, 3)
    assert _crop(image, (1919, 1079)).shape == (96, 96, 3)
