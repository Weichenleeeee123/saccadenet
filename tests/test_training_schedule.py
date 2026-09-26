import pytest

from saccadenet.models.train_fovea import resolve_end_epoch


def test_resume_can_add_a_bounded_number_of_epochs_without_changing_base_config():
    assert resolve_end_epoch(config_epochs=5, start_epoch=5, additional_epochs=3) == 8
    assert resolve_end_epoch(config_epochs=5, start_epoch=0, additional_epochs=0) == 5
    with pytest.raises(ValueError, match="additional_epochs"):
        resolve_end_epoch(config_epochs=5, start_epoch=5, additional_epochs=-1)
