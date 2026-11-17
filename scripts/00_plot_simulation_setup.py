from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from miaa_oct.config import load_config
from miaa_oct.fringe import make_wedge_depths, synthesize_two_reflector_fringe
from miaa_oct.grids import build_grids
from miaa_oct.noise import add_complex_awgn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_quick.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    grids = build_grids(config)
    output = root / "results/figures"
    output.mkdir(parents=True, exist_ok=True)

    z1, z2, separation = make_wedge_depths(
        config.num_a_lines,
        config.center_depth_um * 1e-6,
        config.maximum_separation_um * 1e-6,
        config.minimum_separation_um * 1e-6,
    )
    plt.figure(figsize=(8, 5))
    plt.plot(z1 * 1e6, label="Reflector 1")
    plt.plot(z2 * 1e6, label="Reflector 2")
    plt.xlabel("A-line index")
    plt.ylabel("Depth (µm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "00_wedge_geometry.png", dpi=180)
    plt.close()

    given_mask = np.zeros(config.n_total, dtype=bool)
    given_mask[grids.given_indices] = True
    plt.figure(figsize=(8, 4))
    plt.plot(grids.q_full_rad_m / 1e3, np.ones(config.n_total), linewidth=1)
    plt.fill_between(
        grids.q_full_rad_m / 1e3,
        0,
        1,
        where=given_mask,
        alpha=0.5,
        label="Measured central band",
    )
    plt.fill_between(
        grids.q_full_rad_m / 1e3,
        0,
        1,
        where=~given_mask,
        alpha=0.25,
        label="Missing target band",
    )
    plt.xlabel(r"Centered wavenumber $q=k-k_c$ (krad/m)")
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "00_spectral_regions.png", dpi=180)
    plt.close()

    center = config.center_depth_um * 1e-6
    half = 0.5 * 12e-6
    clean_full = synthesize_two_reflector_fringe(
        grids.q_full_rad_m,
        (center - half, center + half),
        tuple(complex(x) for x in config.reflector_amplitudes),
        carrier_wavenumber_rad_m=grids.carrier_wavenumber_rad_m,
    )
    clean_given = clean_full[grids.given_indices]
    plt.figure(figsize=(9, 6))
    for index, snr_db in enumerate((10.0, 30.0, 50.0), start=1):
        noisy, _, _, _ = add_complex_awgn(
            clean_given,
            snr_db,
            np.random.default_rng(config.random_seed + index),
        )
        axis = np.arange(config.n_given)
        plt.subplot(3, 1, index)
        plt.plot(axis, clean_given.real, label="Clean real fringe", linewidth=1)
        plt.plot(axis, noisy.real, label=f"Noisy, {snr_db:.0f} dB", linewidth=0.8)
        plt.ylabel("Amplitude")
        if index == 1:
            plt.legend(loc="upper right")
    plt.xlabel("Measured spectral-sample index")
    plt.tight_layout()
    plt.savefig(output / "00_noisy_fringe_examples.png", dpi=180)
    plt.close()

    print("Saved setup figures in results/figures/.")


if __name__ == "__main__":
    main()
