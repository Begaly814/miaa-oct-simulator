from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from miaa_oct.config import load_config
from miaa_oct.fringe import make_wedge_depths
from miaa_oct.grids import build_grids
from miaa_oct.pipeline import process_a_line
from miaa_oct.plotting import save_bscan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_quick.yaml")
    parser.add_argument("--snrs", type=float, nargs="*", default=[10.0, 30.0, 50.0])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    grids = build_grids(config)

    z1, z2, separation = make_wedge_depths(
        config.num_a_lines,
        config.center_depth_um * 1e-6,
        config.maximum_separation_um * 1e-6,
        config.minimum_separation_um * 1e-6,
    )

    for snr_db in args.snrs:
        rng = np.random.default_rng(config.random_seed + int(snr_db))
        dft_lines = []
        iaa_lines = []
        miaa_lines = []
        oracle_lines = []
        previous_a = None
        previous_r = None

        for line_index in tqdm(range(config.num_a_lines), desc=f"Wedge {snr_db:.0f} dB"):
            warm = config.use_warm_start and line_index > 0
            result = process_a_line(
                config,
                grids,
                depths_m=(float(z1[line_index]), float(z2[line_index])),
                snr_db=snr_db,
                rng=rng,
                initial_reflectivity=previous_a if warm else None,
                initial_covariance=previous_r if warm else None,
                iterations=config.warm_iterations if warm else config.cold_iterations,
            )
            previous_a = result.iaa.reflectivity
            previous_r = result.iaa.covariance
            dft_lines.append(result.dft_profile)
            iaa_lines.append(result.iaa.reflectivity)
            miaa_lines.append(result.miaa_profile_tapered)
            oracle_lines.append(result.oracle_profile)

        dft_bscan = np.stack(dft_lines)
        iaa_bscan = np.stack(iaa_lines)
        miaa_bscan = np.stack(miaa_lines)
        oracle_bscan = np.stack(oracle_lines)

        prefix = root / "results/figures"
        depth_limits = (config.center_depth_um - 35, config.center_depth_um + 35)
        save_bscan(
            prefix / f"03_wedge_dft_{int(snr_db)}dB.png",
            dft_bscan,
            grids.depth_reconstruction_m,
            depth_limits,
            f"Given-band DFT, {snr_db:.0f} dB",
        )
        save_bscan(
            prefix / f"03_wedge_iaa_{int(snr_db)}dB.png",
            iaa_bscan,
            grids.depth_dictionary_m,
            depth_limits,
            f"IAA, {snr_db:.0f} dB",
        )
        save_bscan(
            prefix / f"03_wedge_miaa_{int(snr_db)}dB.png",
            miaa_bscan,
            grids.depth_reconstruction_m,
            depth_limits,
            f"MIAA, {snr_db:.0f} dB",
        )
        save_bscan(
            prefix / f"03_wedge_oracle_{int(snr_db)}dB.png",
            oracle_bscan,
            grids.depth_reconstruction_m,
            depth_limits,
            "Oracle full-band reference",
        )

        np.savez_compressed(
            root / f"results/wedge_{int(snr_db)}dB.npz",
            z1_m=z1,
            z2_m=z2,
            separation_m=separation,
            depth_dft_m=grids.depth_reconstruction_m,
            depth_iaa_m=grids.depth_dictionary_m,
            dft=dft_bscan,
            iaa=iaa_bscan,
            miaa=miaa_bscan,
            oracle=oracle_bscan,
        )


if __name__ == "__main__":
    main()
