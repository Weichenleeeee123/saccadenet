import pytest
import torch

from saccadenet.cost import accounting
from saccadenet.cost.accounting import CostMeter, count_model_flops
from saccadenet.models.fovea import FoveaNet


def test_single_conv_count_uses_two_flops_per_multiply_accumulate():
    model = torch.nn.Conv2d(1, 2, 3, bias=False)
    assert count_model_flops(model, (1, 1, 5, 5)) == 2 * 3 * 3 * 2 * 3 * 3


def test_fovea_analytic_flops_matches_profile_without_forward():
    assert hasattr(accounting, "count_fovea_flops")
    model = FoveaNet().eval()
    from saccadenet.run.baselines import _DenseAdapter
    expected_patch = count_model_flops(model, (1, 3, 96, 96))
    expected_dense = count_model_flops(_DenseAdapter(model), (1, 3, 129, 161))
    model.forward = lambda *_: (_ for _ in ()).throw(AssertionError("forward called"))
    model.dense = lambda *_: (_ for _ in ()).throw(AssertionError("dense called"))
    assert accounting.count_fovea_flops(model, 96, 96) == expected_patch
    assert accounting.count_fovea_flops(model, 161, 129) == expected_dense


def test_costs_are_monotonic_and_pyramid_is_charged_once():
    meter = CostMeter()
    meter.record_pyramid_once(100)
    meter.add_sensing(10)
    meter.add_semantic("cnn", 200)
    before = meter.snapshot()
    meter.add_semantic("detector", 20)
    after = meter.snapshot()
    assert before["sensing_flops"] == 110
    assert after["semantic_flops"] == 220
    assert after["semantic_flops"] >= before["semantic_flops"]
    with pytest.raises(ValueError, match="pyramid"):
        meter.record_pyramid_once(100)
    with pytest.raises(ValueError, match="negative"):
        meter.add_semantic("cnn", -1)
