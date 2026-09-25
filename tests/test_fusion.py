import math

import numpy as np

from saccadenet.bayes.fusion import FusionB


def test_higher_quality_negative_evidence_replaces_earlier_weak_support():
    fusion = FusionB()
    fusion.update(4, x=0.9, dprime=0.5)
    fusion.update(4, x=0.1, dprime=2.0)
    assert fusion.llr()[4] < 0
    assert math.isclose(fusion.llr()[4], 4 * (0.1 - 0.5))


def test_repeated_or_lower_quality_view_cannot_multiply_evidence():
    fusion = FusionB()
    fusion.update(2, x=0.8, dprime=1.0)
    first = fusion.llr()[2]
    fusion.update(2, x=0.8, dprime=1.0)
    fusion.update(2, x=1.0, dprime=0.2)
    assert fusion.llr()[2] == first
    assert fusion.gain(2, 1.0) == 0


def test_new_candidate_is_added_without_erasing_old_evidence():
    fusion = FusionB()
    fusion.update(1, x=1.0, dprime=2.0)
    first = fusion.llr()[1]
    before = fusion.posterior([1, 2])
    fusion.update(3, x=0.5, dprime=1.0)
    after = fusion.posterior([1, 2, 3])
    assert fusion.llr()[1] == first
    assert len(after) == 3
    assert np.isclose(sum(after), 1.0)
    assert after[0] < before[0]


def test_extreme_logits_have_finite_posterior_and_reset_clears_state():
    fusion = FusionB()
    fusion.update_llr(1, 10000.0, quality=1.0)
    fusion.update_llr(2, -10000.0, quality=1.0)
    assert np.all(np.isfinite(fusion.posterior([1, 2])))
    fusion.reset(1)
    assert fusion.llr()[1] == 0
