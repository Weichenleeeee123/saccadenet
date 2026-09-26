"""Render backup slide images of an exported demo with headless Edge/Chrome (offline, file:// only)."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image

BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)
DEFAULT_SHOTS = ("0:1:view-seen", "0:18:explore", "0:30:found", "1:25:24k-frozen-miss", "2:65:24k-derived-hit", "3:2:overconfident-fail")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True, help="artifacts/demo/demo-* directory")
    parser.add_argument("--shot", action="append", help="episode_index:step:name (1-based step)")
    parser.add_argument("--out", type=Path, default=Path("reports/replay"))
    args = parser.parse_args()
    browser = next((path for path in BROWSERS if Path(path).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if browser is None:
        raise SystemExit("no Edge/Chrome found")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = args.out / f"snapshots-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    page = (args.demo.resolve() / "index.html").as_uri()
    with tempfile.TemporaryDirectory(dir=args.demo.parent) as profile:
        for spec in args.shot or DEFAULT_SHOTS:
            episode, step, name = spec.split(":", 2)
            png = Path(profile) / f"{name}.png"
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1600,1100",
                            "--virtual-time-budget=6000", f"--user-data-dir={profile}", f"--screenshot={png}",
                            f"{page}?ep={episode}&step={step}"], check=True, capture_output=True)
            Image.open(png).convert("RGB").save(target / f"{name}.jpg", quality=85, optimize=True)
    (target / "source.txt").write_text(f"demo: {args.demo}\nshots: {list(args.shot or DEFAULT_SHOTS)}\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
