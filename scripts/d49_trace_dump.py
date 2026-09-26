"""D49: dump full episode traces on development canvases for bit-identity comparison."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from saccadenet.bayes.calibrate import GaussianCalibrator
from saccadenet.config import EpisodeConfig
from saccadenet.contracts import EpisodeInput
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.models.fovea import FoveaNet
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.run.episode import SceneSensor, run_episode

PLAN = ((1920, 1080, 20), (3840, 2160, 10), (7680, 4320, 6), (15360, 8640, 4))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return repr(value)
    return value


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
cfg = yaml.safe_load(Path("configs/recovery_final_b04.yaml").read_text(encoding="utf-8"))
device = torch.device("cuda")
net = FoveaNet().to(device).eval()
net.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)["model"])
cal = GaussianCalibrator(**json.loads(Path(cfg["retina_fit"]).read_text(encoding="utf-8"))["model"])
bank = MnistBank.from_split("development")
digests = {}
for width, height, count in PLAN:
    for seed in range(21100, 21100 + count):
        config = EpisodeConfig(width=width, height=height, query=seed % 10)
        image, _ = make_canvas(config, seed, bank)
        sensor = SceneSensor(build_pyramid(image), fovea_size=config.fovea_size, sectors=config.sectors)
        log = run_episode(sensor, EpisodeInput(config.query, width, height, config.k), config, cal, net, device=device)
        record = plain({"answer": log.answer_xy, "reason": log.reason, "steps": log.steps, "trace": log.trace, "costs": log.costs})
        text = json.dumps(record, sort_keys=True, ensure_ascii=False)
        key = f"{width}x{height}-{seed}"
        (out / f"{key}.json").write_text(text, encoding="utf-8")
        digests[key] = hashlib.sha256(text.encode()).hexdigest()
        print(key, log.steps, log.reason, digests[key][:12], flush=True)
(out / "digests.json").write_text(json.dumps(digests, indent=1), encoding="utf-8")
