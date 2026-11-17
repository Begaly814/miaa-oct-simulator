from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from miaa_oct.config import load_config
from miaa_oct.grids import build_grids
from miaa_oct.metrics import evaluate_two_reflector_resolution, normalized_mse_db
from miaa_oct.pipeline import process_a_line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_quick.yaml")
    parser.add_argument("--separation-step-um", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    grids = build_grids(config)
    separations_um = np.arange(
        0.0,
        config.maximum_separation_um + 0.5 * args.separation_step_um,
        args.separation_step_um,
    )
    rows: list[dict[str, float | int | bool | str]] = []

    for snr_db in config.snr_levels_db:
        for seed_index in tqdm(
            range(config.monte_carlo_seeds),
            desc=f"SNR {snr_db:.0f} dB",
        ):
            rng = np.random.default_rng(config.random_seed + 1000 * int(snr_db) + seed_index)
            previous_a = None
            previous_r = None

            for separation_um in separations_um[::-1]:
                center = config.center_depth_um * 1e-6
                half = 0.5 * separation_um * 1e-6
                true_depths = (center - half, center + half)
                warm = config.use_warm_start and previous_a is not None

                result = process_a_line(
                    config,
                    grids,
                    depths_m=true_depths,
                    snr_db=float(snr_db),
                    rng=rng,
                    initial_reflectivity=previous_a if warm else None,
                    initial_covariance=previous_r if warm else None,
                    iterations=config.warm_iterations if warm else config.cold_iterations,
                )
                previous_a = result.iaa.reflectivity
                previous_r = result.iaa.covariance

                method_profiles = {
                    "DFT": (result.depth_reconstruction_m, result.dft_profile),
                    "IAA": (grids.depth_dictionary_m, result.iaa.reflectivity),
                    "MIAA_raw": (result.depth_reconstruction_m, result.miaa_profile_raw),
                    "MIAA_tapered": (
                        result.depth_reconstruction_m,
                        result.miaa_profile_tapered,
                    ),
                }
                missing_nmse_db = normalized_mse_db(
                    result.miaa.missing_spectrum,
                    result.clean_full_spectrum[grids.missing_indices],
                )

                for method, (depth_axis, profile) in method_profiles.items():
                    metric = evaluate_two_reflector_resolution(
                        depth_axis,
                        profile,
                        true_depths,
                        valley_fraction=config.resolution_valley_fraction,
                        search_window_m=config.peak_search_window_um * 1e-6,
                        localization_tolerance_m=config.localization_tolerance_um * 1e-6,
                    )
                    rows.append(
                        {
                            "snr_db": float(snr_db),
                            "measured_snr_db": result.measured_snr_db,
                            "seed": seed_index,
                            "separation_um": float(separation_um),
                            "method": method,
                            "resolved": metric.resolved,
                            "localization_rmse_um": (
                                np.nan
                                if metric.localization_rmse_m is None
                                else metric.localization_rmse_m * 1e6
                            ),
                            "amplitude_ratio_error_db": (
                                np.nan
                                if metric.amplitude_ratio_error_db is None
                                else metric.amplitude_ratio_error_db
                            ),
                            "missing_spectrum_nmse_db": (
                                missing_nmse_db if method.startswith("MIAA") else np.nan
                            ),
                        }
                    )

    output = root / "results/tables/snr_sweep.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
