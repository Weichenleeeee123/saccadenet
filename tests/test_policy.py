import numpy as np

import saccadenet.bayes.policy as policy
from saccadenet.bayes.fusion import FusionB
from saccadenet.bayes.policy import choose_fixation, exploration_anchor, fixation_candidates, should_stop
from saccadenet.contracts import Candidate


def _candidate(stable_id, x, y):
    return Candidate(stable_id, (x, y), 1.0, 0, 0)


def test_two_visible_candidates_can_make_midpoint_best():
    candidates = [_candidate(0, 100, 100), _candidate(1, 140, 100)]
    omega = fixation_candidates(candidates, e_half=30, canvas_shape=(200, 200))
    assert (120.0, 100.0) in omega
    chosen = choose_fixation(
        np.asarray((0.5, 0.5)),
        candidates,
        FusionB(),
        omega,
        current=(0, 0),
        dprime=lambda e: 1.0 if e <= 25 else 0.0,
    )
    assert chosen == (120.0, 100.0)


def test_map_policy_chooses_largest_posterior():
    candidates = [_candidate(0, 50, 50), _candidate(1, 90, 90)]
    chosen = choose_fixation(
        np.asarray((0.2, 0.8)), candidates, FusionB(), [c.xy for c in candidates],
        current=(0, 0), dprime=lambda e: 1.0, strategy="map"
    )
    assert chosen == (90, 90)


def test_fixed_policy_cycles_candidate_options_without_using_posterior():
    candidates = [_candidate(0, 50, 50), _candidate(1, 90, 90)]
    options = [candidate.xy for candidate in candidates]
    posterior = np.asarray((0.01, 0.99))
    assert choose_fixation(
        posterior, candidates, FusionB(), options,
        current=(0, 0), dprime=lambda e: 1.0, strategy="fixed", step=0,
    ) == (50, 50)
    assert choose_fixation(
        posterior, candidates, FusionB(), options,
        current=(0, 0), dprime=lambda e: 1.0, strategy="fixed", step=1,
    ) == (90, 90)


def test_random_policy_replays_with_same_seed():
    candidates = [_candidate(0, 50, 50), _candidate(1, 90, 90)]
    options = [candidate.xy for candidate in candidates]
    first = choose_fixation(
        np.asarray((0.5, 0.5)), candidates, FusionB(), options,
        current=(0, 0), dprime=lambda e: 1.0, strategy="random",
        rng=np.random.default_rng(123),
    )
    again = choose_fixation(
        np.asarray((0.9, 0.1)), candidates, FusionB(), options,
        current=(0, 0), dprime=lambda e: 1.0, strategy="random",
        rng=np.random.default_rng(123),
    )
    assert first == again
    assert first in options


def test_no_candidate_uses_fixed_grid_then_exhausts():
    visited = set()
    first = exploration_anchor(500, 300, grid=5, visited=visited, current=(250, 150))
    assert first == (250.0, 150.0)
    for row in range(5):
        for col in range(5):
            visited.add(((col + 0.5) * 500 / 5, (row + 0.5) * 300 / 5))
    assert exploration_anchor(500, 300, grid=5, visited=visited, current=(250, 150)) is None


def test_single_discovered_candidate_cannot_force_threshold_stop():
    assert not should_stop(np.asarray((1.0,)), discovered=1, expected_k=12, tau=0.95, has_evidence=True)
    assert not should_stop(np.asarray((0.97, 0.03)), discovered=12, expected_k=12, tau=0.95, has_evidence=False)
    assert should_stop(np.asarray((0.96, 0.04)), discovered=12, expected_k=12, tau=0.95, has_evidence=True)


def test_confirmation_requires_every_discovered_candidate_to_have_evidence():
    posterior = np.asarray((0.99, 0.01))
    assert not should_stop(
        posterior, discovered=2, expected_k=2, tau=0.95,
        has_evidence=True, require_all_scored=True, all_scored=False,
    )
    assert should_stop(
        posterior, discovered=2, expected_k=2, tau=0.95,
        has_evidence=True, require_all_scored=True, all_scored=True,
    )


def test_confirmation_targets_nearest_unscored_candidate():
    candidates = [_candidate(0, 10, 10), _candidate(1, 80, 80), _candidate(2, 25, 25)]
    fusion = FusionB()
    fusion.update(0, 1.0, 2.0)
    assert policy.confirmation_fixation(candidates, fusion, current=(0, 0)) == (25, 25)
    fusion.update(2, 0.0, 2.0)
    assert policy.confirmation_fixation(candidates, fusion, current=(0, 0)) == (80, 80)
    fusion.update(1, 0.0, 2.0)
    assert policy.confirmation_fixation(candidates, fusion, current=(0, 0)) is None
