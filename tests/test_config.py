import pytest

from saccadenet.config import EpisodeConfig, config_hash, config_from_mapping


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"k": -1}, "k"),
        ({"width": 2000}, "resolution"),
        ({"tau": 0}, "tau"),
        ({"tau": 1}, "tau"),
        ({"t_max": 0}, "t_max"),
        ({"detector_threshold": 256}, "detector_threshold"),
        ({"candidate_merge_radius": 0}, "candidate_merge_radius"),
        ({"exploration_grid": 0}, "exploration_grid"),
    ],
)
def test_invalid_episode_config_names_bad_field(overrides, field):
    with pytest.raises(ValueError, match=field):
        EpisodeConfig(**overrides)


def test_config_hash_ignores_mapping_order():
    first = config_from_mapping({"height": 1080, "width": 1920, "k": 12})
    second = config_from_mapping({"k": 12, "width": 1920, "height": 1080})
    assert config_hash(first) == config_hash(second)


def test_config_rejects_unrecognized_override():
    with pytest.raises(ValueError, match="unknown"):
        config_from_mapping({"unknown": 1})
