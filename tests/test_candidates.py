import numpy as np

from saccadenet.retina.detect import CandidateTracker, Detection, detect_on_retina
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import sample_retina


def _observe(cards, fixation=(256, 256)):
    image = np.full((512, 512, 3), 126, dtype=np.uint8)
    for x, y, half in cards:
        image[y - half : y + half, x - half : x + half] = 225
    return sample_retina(build_pyramid(image), fixation)


def test_empty_scene_has_no_candidates():
    assert detect_on_retina(_observe([])) == []


def test_card_crossing_angular_seam_is_detected():
    found = detect_on_retina(_observe([(448, 256, 48)]))
    assert any(np.linalg.norm(np.asarray(item.xy) - (448, 256)) <= 48 for item in found)


def test_two_separated_cards_are_not_collapsed():
    found = detect_on_retina(_observe([(112, 200, 48), (400, 200, 48)]))
    for expected in ((112, 200), (400, 200)):
        assert any(np.linalg.norm(np.asarray(item.xy) - expected) <= 48 for item in found)


def test_repeated_observation_keeps_candidate_identity():
    tracker = CandidateTracker(merge_radius=96, reset_distance=8)
    first, reset = tracker.update([Detection((200.0, 120.0), 0.7)], step=0)
    assert len(first) == 1
    assert reset == set()
    second, reset = tracker.update([Detection((203.0, 121.0), 0.8)], step=1)
    assert len(second) == 1
    assert second[0].stable_id == first[0].stable_id
    assert reset == set()
    _, reset = tracker.update([Detection((220.0, 120.0), 0.9)], step=2)
    assert reset == {first[0].stable_id}


def test_weaker_edge_detection_does_not_drag_a_better_center():
    tracker = CandidateTracker(merge_radius=144)
    tracker.update([Detection((11330.4, 3771.8), 0.82)], step=0)
    tracker.update([Detection((11356.3, 3752.4), 0.83)], step=1)
    candidates, _ = tracker.update([Detection((11315.6, 3798.0), 0.77)], step=2)
    assert candidates[0].xy == (11356.3, 3752.4)


def test_distant_detections_remain_distinct():
    tracker = CandidateTracker(merge_radius=96)
    candidates, _ = tracker.update(
        [Detection((90.0, 90.0), 1.0), Detection((390.0, 90.0), 1.0)], step=0
    )
    assert len(candidates) == 2
