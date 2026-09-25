"""Export one retina observation and candidate reconstruction for visual audit."""

from pathlib import Path

import numpy as np
from PIL import Image

from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import make_canvas
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.reconstruct import candidate_view
from saccadenet.retina.sampler import sample_retina


class SyntheticDigitBank:
    def sample(self, label, rng):
        digit = np.zeros((28, 28), dtype=np.uint8)
        digit[6:22, 8:20] = 220
        return digit, int(rng.integers(0, 1_000_000))


def main() -> None:
    root = Path("reports/spikes")
    root.mkdir(parents=True, exist_ok=True)
    config = EpisodeConfig()
    canvas, truth = make_canvas(config, 123, SyntheticDigitBank())
    fixation = (config.width / 2, config.height / 2)
    retina = sample_retina(build_pyramid(canvas), fixation)
    candidate = tuple(float(x) for x in truth.centers_xy[truth.target_index])
    reconstruction, valid = candidate_view(retina, candidate)
    Image.fromarray(retina.fovea).save(root / "retina_preview_fovea.png")
    Image.fromarray(retina.logpolar).save(root / "retina_preview_strip.png")
    Image.fromarray(reconstruction).save(root / "retina_preview_reconstruction.png")
    Image.fromarray((valid * 255).astype(np.uint8)).save(root / "retina_preview_valid.png")
    print(f"candidate={candidate} fixation={fixation} valid_fraction={valid.mean():.3f}")


if __name__ == "__main__":
    main()
