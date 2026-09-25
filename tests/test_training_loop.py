import torch

from saccadenet.models.fovea import FoveaNet
from saccadenet.models.train_fovea import train_one_batch


def test_repeated_training_on_one_batch_reduces_cross_entropy():
    torch.manual_seed(22)
    model = FoveaNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    images = torch.rand(4, 3, 96, 96)
    labels = torch.tensor([0, 1, 2, 3])
    first = train_one_batch(model, optimizer, images, labels, device=torch.device("cpu"))
    last = first
    for _ in range(5):
        last = train_one_batch(model, optimizer, images, labels, device=torch.device("cpu"))
    assert last < first
