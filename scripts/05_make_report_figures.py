from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def first_merge_resolution(group: pd.DataFrame, probability: float = 0.90) -> float:
    """Resolution threshold when sweeping from large to small separation.

    Returns the last separation before the success probability first falls below the
    chosen probability. This is less vulnerable to isolated nonmonotonic points than
    simply taking the smallest successful separation.
    """
    ordered = group.sort_values("separation_um", ascending=False)
    passed = ordered["probability_resolved"].to_numpy() >= probability
    separations = ordered["separation_um"].to_numpy()
    if not passed.any():
        return float("nan")
    failing = np.flatnonzero(~passed)
    if failing.size == 0:
        return float(np.min(separations))
    first_failure = int(failing[0])
    if first_failure == 0:
        return float("nan")
    return float(separations[first_failure - 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/tables/snr_sweep.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / args.input)
    figure_dir = root / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        data.groupby(["snr_db", "separation_um", "method"], as_index=False)
        .agg(
            probability_resolved=("resolved", "mean"),
            localization_rmse_um=("localization_rmse_um", "mean"),
            amplitude_ratio_error_db=("amplitude_ratio_error_db", "mean"),
            missing_spectrum_nmse_db=("missing_spectrum_nmse_db", "mean"),
        )
    )
    summary.to_csv(root / "results/tables/snr_sweep_summary.csv", index=False)

    threshold_rows = []
    for (snr_db, method), group in summary.groupby(["snr_db", "method"]):
        threshold_rows.append(
            {
                "snr_db": snr_db,
                "method": method,
                "resolution_90_um": first_merge_resolution(group, probability=0.90),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(root / "results/tables/resolution_90.csv", index=False)

    plt.figure(figsize=(7, 5))
    for method, group in thresholds.groupby("method"):
        plt.plot(group["snr_db"], group["resolution_90_um"], marker="o", label=method)
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("90% two-reflector resolution (µm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "05_resolution_vs_snr.png", dpi=180)
    plt.close()

    miaa = summary[summary["method"] == "MIAA_raw"]
    nmse = miaa.groupby("snr_db", as_index=False)["missing_spectrum_nmse_db"].mean()
    plt.figure(figsize=(7, 5))
    plt.plot(nmse["snr_db"], nmse["missing_spectrum_nmse_db"], marker="o")
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Missing-spectrum NMSE (dB)")
    plt.tight_layout()
    plt.savefig(figure_dir / "05_missing_nmse_vs_snr.png", dpi=180)
    plt.close()

    heat = summary[summary["method"] == "MIAA_tapered"].pivot(
        index="separation_um",
        columns="snr_db",
        values="probability_resolved",
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(
        heat.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[heat.columns.min(), heat.columns.max(), heat.index.min(), heat.index.max()],
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plt.colorbar(label="Probability resolved")
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Reflector separation (µm)")
    plt.tight_layout()
    plt.savefig(figure_dir / "05_resolution_probability_heatmap.png", dpi=180)
    plt.close()

    print("Saved report figures and summary tables.")


if __name__ == "__main__":
    main()
