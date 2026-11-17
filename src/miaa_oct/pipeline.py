from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .fringe import synthesize_two_reflector_fringe
from .grids import GridSet
from .iaa import IAAResult, iaa_reconstruct
from .miaa import MIAAResult, miaa_extrapolate
from .noise import add_complex_awgn
from .reconstruction import (
    given_band_dft_spectrum,
    reconstruct_from_uniform_q,
    squared_cosine_edge_taper,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ALineResult:
    clean_full_spectrum: ComplexArray
    clean_given_spectrum: ComplexArray
    noisy_given_spectrum: ComplexArray
    measured_snr_db: float
    noise_variance: float
    iaa: IAAResult
    miaa: MIAAResult
    depth_reconstruction_m: FloatArray
    dft_profile: ComplexArray
    miaa_profile_raw: ComplexArray
    miaa_profile_tapered: ComplexArray
    oracle_profile: ComplexArray


def process_a_line(
    config: SimulationConfig,
    grids: GridSet,
    depths_m: tuple[float, float],
    snr_db: float,
    rng: np.random.Generator,
    initial_reflectivity: ComplexArray | None = None,
    initial_covariance: ComplexArray | None = None,
    iterations: int | None = None,
) -> ALineResult:
    clean_full = synthesize_two_reflector_fringe(
        grids.q_full_rad_m,
        depths_m,
        tuple(complex(x) for x in config.reflector_amplitudes),
        carrier_wavenumber_rad_m=grids.carrier_wavenumber_rad_m,
    )
    clean_given = clean_full[grids.given_indices]
    noisy_given, _, noise_variance, measured_snr = add_complex_awgn(
        clean_given,
        snr_db,
        rng,
        exact_realization_snr=True,
    )

    iaa_result = iaa_reconstruct(
        noisy_given,
        grids.f_given,
        noise_variance=noise_variance,
        iterations=iterations or config.cold_iterations,
        diagonal_loading=config.diagonal_loading,
        initial_reflectivity=initial_reflectivity,
        initial_covariance=initial_covariance,
    )
    miaa_result = miaa_extrapolate(
        noisy_given,
        iaa_result.reflectivity,
        iaa_result.covariance,
        grids.f_given,
        grids.f_missing,
        grids.given_indices,
        grids.missing_indices,
        config.n_total,
    )

    dft_spectrum = given_band_dft_spectrum(
        noisy_given,
        grids.given_indices,
        config.n_total,
    )
    depth, dft_profile = reconstruct_from_uniform_q(
        dft_spectrum,
        grids.dq_rad_m,
        config.n_recon,
    )
    _, miaa_raw = reconstruct_from_uniform_q(
        miaa_result.full_spectrum,
        grids.dq_rad_m,
        config.n_recon,
    )
    taper = squared_cosine_edge_taper(config.n_total, config.edge_taper_fraction)
    _, miaa_tapered = reconstruct_from_uniform_q(
        miaa_result.full_spectrum * taper,
        grids.dq_rad_m,
        config.n_recon,
    )
    _, oracle = reconstruct_from_uniform_q(
        clean_full,
        grids.dq_rad_m,
        config.n_recon,
    )

    return ALineResult(
        clean_full_spectrum=clean_full,
        clean_given_spectrum=clean_given,
        noisy_given_spectrum=noisy_given,
        measured_snr_db=measured_snr,
        noise_variance=noise_variance,
        iaa=iaa_result,
        miaa=miaa_result,
        depth_reconstruction_m=depth,
        dft_profile=dft_profile,
        miaa_profile_raw=miaa_raw,
        miaa_profile_tapered=miaa_tapered,
        oracle_profile=oracle,
    )
