from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from miaa_oct.config import load_config
from miaa_oct.fringe import synthesize_reflector_fringe
from miaa_oct.grids import build_grids
from miaa_oct.iaa import iaa_reconstruct
from miaa_oct.metrics import normalized_mse_db
from miaa_oct.miaa import miaa_extrapolate
from miaa_oct.noise import add_complex_awgn
from miaa_oct.plotting import save_profiles_with_axes
from miaa_oct.reconstruction import reconstruct_from_uniform_q, squared_cosine_edge_taper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_quick.yaml")
    parser.add_argument("--snr", type=float, default=60.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    grids = build_grids(config)
    rng = np.random.default_rng(config.random_seed)

    z = config.center_depth_um * 1e-6
    clean_full = synthesize_reflector_fringe(
        grids.q_full_rad_m,
        [z],
        [1.0 + 0j],
        carrier_wavenumber_rad_m=grids.carrier_wavenumber_rad_m,
    )
    clean_given = clean_full[grids.given_indices]
    noisy_given, _, noise_variance, measured_snr = add_complex_awgn(
        clean_given,
        args.snr,
        rng,
    )

    iaa = iaa_reconstruct(
        noisy_given,
        grids.f_given,
        noise_variance,
        iterations=config.cold_iterations,
        diagonal_loading=config.diagonal_loading,
    )
    miaa = miaa_extrapolate(
        noisy_given,
        iaa.reflectivity,
        iaa.covariance,
        grids.f_given,
        grids.f_missing,
        grids.given_indices,
        grids.missing_indices,
        config.n_total,
    )
    depth, miaa_profile = reconstruct_from_uniform_q(
        miaa.full_spectrum * squared_cosine_edge_taper(config.n_total, config.edge_taper_fraction),
        grids.dq_rad_m,
        config.n_recon,
    )
    _, oracle = reconstruct_from_uniform_q(clean_full, grids.dq_rad_m, config.n_recon)

    nmse_db = normalized_mse_db(
        miaa.missing_spectrum,
        clean_full[grids.missing_indices],
    )
    peak_index = int(np.argmax(np.abs(iaa.reflectivity)))
    estimated_depth_um = grids.depth_dictionary_m[peak_index] * 1e6

    print(f"Measured input SNR: {measured_snr:.2f} dB")
    print(f"IAA estimated depth: {estimated_depth_um:.3f} µm")
    print(f"True depth: {z * 1e6:.3f} µm")
    print(f"Missing-spectrum NMSE: {nmse_db:.2f} dB")
    print(f"IAA iterations: {iaa.iterations}")

    save_profiles_with_axes(
        root / "results/figures/01_single_reflector.png",
        {
            "IAA depth estimate": (grids.depth_dictionary_m, iaa.reflectivity),
            "MIAA": (depth, miaa_profile),
            "Oracle full band": (depth, oracle),
        },
        true_depths_m=(z,),
        x_limits_um=(config.center_depth_um - 30, config.center_depth_um + 30),
    )
    print("Saved results/figures/01_single_reflector.png")


if __name__ == "__main__":
    main()
