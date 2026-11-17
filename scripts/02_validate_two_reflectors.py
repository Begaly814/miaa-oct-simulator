from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from miaa_oct.config import load_config
from miaa_oct.grids import build_grids
from miaa_oct.metrics import evaluate_two_reflector_resolution, normalized_mse_db
from miaa_oct.pipeline import process_a_line
from miaa_oct.plotting import save_profiles_with_axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_quick.yaml")
    parser.add_argument("--snr", type=float, default=50.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    grids = build_grids(config)
    rng = np.random.default_rng(config.random_seed + 1)

    for separation_um in (25.0, 7.0):
        center = config.center_depth_um * 1e-6
        half = 0.5 * separation_um * 1e-6
        true_depths = (center - half, center + half)
        result = process_a_line(
            config,
            grids,
            depths_m=true_depths,
            snr_db=args.snr,
            rng=rng,
        )

        method_profiles = {
            "DFT": (result.depth_reconstruction_m, result.dft_profile),
            "IAA": (grids.depth_dictionary_m, result.iaa.reflectivity),
            "MIAA": (result.depth_reconstruction_m, result.miaa_profile_tapered),
        }
        messages = []
        for method, (depth_axis, profile) in method_profiles.items():
            metric = evaluate_two_reflector_resolution(
                depth_axis,
                profile,
                true_depths,
                valley_fraction=config.resolution_valley_fraction,
                search_window_m=config.peak_search_window_um * 1e-6,
                localization_tolerance_m=config.localization_tolerance_um * 1e-6,
            )
            messages.append(f"{method} resolved={metric.resolved}")

        nmse = normalized_mse_db(
            result.miaa.missing_spectrum,
            result.clean_full_spectrum[grids.missing_indices],
        )
        print(
            f"Separation {separation_um:.1f} µm | "
            + " | ".join(messages)
            + f" | missing NMSE={nmse:.2f} dB"
        )

        save_profiles_with_axes(
            root / f"results/figures/02_two_reflectors_{separation_um:.0f}um.png",
            {
                "Given-band DFT": (result.depth_reconstruction_m, result.dft_profile),
                "IAA": (grids.depth_dictionary_m, result.iaa.reflectivity),
                "MIAA": (result.depth_reconstruction_m, result.miaa_profile_tapered),
                "Oracle full band": (result.depth_reconstruction_m, result.oracle_profile),
            },
            true_depths_m=true_depths,
            x_limits_um=(config.center_depth_um - 35, config.center_depth_um + 35),
        )


if __name__ == "__main__":
    main()
