"""Small shared 11-class CNN with a dense sliding-window form."""

import torch
from torch import nn


class FoveaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        stages = []
        input_channels = 3
        for output_channels, kernel in ((16, 5), (32, 3), (64, 3), (64, 3)):
            stages.extend((nn.Conv2d(input_channels, output_channels, kernel), nn.ReLU(), nn.MaxPool2d(2)))
            input_channels = output_channels
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Conv2d(64, 11, 4)

    def dense(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image))

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 4 or patch.shape[1:] != (3, 96, 96):
            raise ValueError("patch must be NCHW with shape N×3×96×96")
        return self.dense(patch).flatten(1)

