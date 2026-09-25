"""Per-component operation estimates with no duplicate pyramid charge."""

from collections import defaultdict

import torch


class CostMeter:
    def __init__(self) -> None:
        self.sensing_flops = 0
        self.sensing_bytes = 0
        self._semantic: dict[str, int] = defaultdict(int)
        self._pyramid_charged = False

    def record_pyramid_once(self, estimated_flops: int) -> None:
        if self._pyramid_charged:
            raise ValueError("pyramid already charged")
        self.add_sensing(estimated_flops)
        self._pyramid_charged = True

    def add_sensing(self, estimated_flops: int) -> None:
        if estimated_flops < 0:
            raise ValueError("negative sensing cost")
        self.sensing_flops += int(estimated_flops)

    def add_sensing_bytes(self, bytes_read: int) -> None:
        if bytes_read < 0:
            raise ValueError("negative sensing bytes")
        self.sensing_bytes += int(bytes_read)

    def add_semantic(self, component: str, estimated_flops: int) -> None:
        if estimated_flops < 0:
            raise ValueError("negative semantic cost")
        self._semantic[component] += int(estimated_flops)

    def snapshot(self) -> dict[str, int]:
        return {
            "sensing_flops": self.sensing_flops,
            "sensing_bytes": self.sensing_bytes,
            "semantic_flops": sum(self._semantic.values()),
            **{f"semantic_{key}_flops": value for key, value in self._semantic.items()},
        }


def count_model_flops(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """Count Conv2d/Linear MACs as 2 FLOPs; unsupported ops excluded."""
    result = 0
    hooks = []

    def count(layer, _inputs, output):
        nonlocal result
        if isinstance(layer, torch.nn.Conv2d):
            kernel = layer.kernel_size[0] * layer.kernel_size[1]
            result += output.numel() * (layer.in_channels // layer.groups) * kernel * 2
        elif isinstance(layer, torch.nn.Linear):
            result += output.numel() * layer.in_features * 2

    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(layer.register_forward_hook(count))
    device = next(model.parameters(), torch.empty(0)).device
    try:
        with torch.inference_mode():
            model(torch.zeros(input_shape, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    return result
