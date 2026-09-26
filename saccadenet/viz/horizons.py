"""Figure 7: the two horizons — digit readability d'(e) vs card detectability P(e) on one axis."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from saccadenet.retina.horizon import detection_horizon


def dprime_bins(fit: dict) -> list[dict]:
    model = fit["model"]
    rows = []
    for index, (low, high) in enumerate(zip(model["edges"][:-1], model["edges"][1:])):
        sigma = model["sigma"][index]
        rows.append({
            "ecc_low": low, "ecc_high": high, "mu0": model["mu0"][index], "mu1": model["mu1"][index], "sigma": sigma,
            "dprime": (model["mu1"][index] - model["mu0"][index]) / sigma,
            "n_target": model["count1"][index], "n_nontarget": model["count0"][index],
        })
    return rows


def render(calibration: Path, horizon_csv: Path, out: Path) -> Path:
    fit = json.loads(calibration.read_text(encoding="utf-8"))
    bins = dprime_bins(fit)
    with horizon_csv.open(newline="", encoding="utf-8") as file:
        detect = [row for row in csv.DictReader(file) if int(row["exposures"]) > 0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"two-horizons-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "dprime-bins.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(bins[0]))
        writer.writeheader()
        writer.writerows(bins)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False})
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.2, top=0.88)
    mid = lambda low, high: np.sqrt(max(low, 6.0) * high)  # geometric bin center on a log axis
    x_d = [mid(b["ecc_low"], b["ecc_high"]) for b in bins]
    y_d = [b["dprime"] for b in bins]
    ax.plot(x_d, y_d, marker="s", color="#d1495b", label="Digit readability d'(e) (calibration split)")
    ax.axhline(1, color="#d1495b", linestyle=":", linewidth=1)
    ax.set_ylabel("d' (query vs other digits)", color="#d1495b")
    ax.set_xscale("log")
    ax.set_xlim(8, 20000)
    ax.set_ylim(-0.2, 4.8)
    twin = ax.twinx()
    x_p = [mid(float(r["ecc_low"]), float(r["ecc_high"])) for r in detect]
    y_p = [float(r["p_discover"]) for r in detect]
    twin.plot(x_p, y_p, marker="o", color="#006d77", label="Card detection P(e) (E1 traces)")
    twin.set_ylim(-0.04, 1.0 * 4.8 / 4.6)
    twin.set_ylabel("P(card discovered at this glimpse)", color="#006d77")
    r1, r2 = detection_horizon()
    twin.axvline(r1, color="#006d77", linestyle="--", linewidth=1)
    twin.axvline(r2, color="#006d77", linestyle=":", linewidth=1)
    twin.text(r1 * 1.05, 0.55, f"r_where = {r1:.0f} px\n(analytic)", color="#006d77", fontsize=8)
    ax.axvspan(8, 144, color="#d1495b", alpha=0.07, lw=0)
    ax.text(10, 0.25, "r_what ~ 150 px:\ndigits readable (d' > 1)", color="#d1495b", fontsize=8, va="bottom")
    ax.set_xlabel("Eccentricity from fixation (px)")
    ax.set_title("Figure 7. Two horizons: what (~150 px) vs where (~1,844 px)")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    fig.text(0.09, 0.07, f"d' bins: {calibration.name} (250 calibration scenes; last bin 972-2000 px).", fontsize=7)
    fig.text(0.09, 0.035, f"Detection: {horizon_csv.parent.name}/horizon.csv", fontsize=7)
    fig.savefig(target / "figure7-two-horizons.png", dpi=180)
    fig.savefig(target / "figure7-two-horizons.svg")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--horizon", type=Path, required=True, help="horizon.csv from saccadenet.exp.trace_analysis")
    parser.add_argument("--out", type=Path, default=Path("reports/analysis"))
    args = parser.parse_args()
    print(render(args.calibration, args.horizon, args.out))


if __name__ == "__main__":
    main()
