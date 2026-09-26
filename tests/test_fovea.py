import io

import torch

from saccadenet.models.fovea import FoveaNet
from saccadenet.models.train_fovea import load_training_state, save_training_state


def test_patch_and_dense_outputs_have_correct_shapes():
    network = FoveaNet()
    assert network(torch.rand(1, 3, 96, 96)).shape == (1, 11)
    assert network(torch.rand(4, 3, 96, 96)).shape == (4, 11)
    assert network.dense(torch.rand(1, 3, 128, 160)).shape == (1, 11, 3, 5)


def test_dense_output_matches_corresponding_patch_at_stride_16():
    torch.manual_seed(5)
    network = FoveaNet().eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.inference_mode():
        dense = network.dense(image)
        patch = network(image[:, :, 16:112, 32:128])
    assert torch.allclose(dense[0, :, 1, 2], patch[0], atol=1e-6)


def test_optimizer_state_is_restored_from_in_memory_checkpoint():
    torch.manual_seed(3)
    network = FoveaNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    x = torch.rand(2, 3, 96, 96)
    loss = network(x).sum()
    loss.backward()
    optimizer.step()
    buffer = io.BytesIO()
    save_training_state(buffer, network, optimizer, epoch=2, config_hash="abc")
    buffer.seek(0)
    restored = FoveaNet()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    epoch, config_hash = load_training_state(buffer, restored, restored_optimizer)
    assert (epoch, config_hash) == (2, "abc")
    for before, after in zip(network.parameters(), restored.parameters()):
        assert torch.equal(before, after)
    assert restored_optimizer.state_dict()["state"]
