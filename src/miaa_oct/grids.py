from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class GridSet:
    carrier_wavenumber_rad_m: float
    q_full_rad_m: FloatArray
    q_given_rad_m: FloatArray
    q_missing_rad_m: FloatArray
    given_indices: NDArray[np.int64]
    missing_indices: NDArray[np.int64]
    depth_dictionary_m: FloatArray
    depth_reconstruction_m: FloatArray
    f_given: ComplexArray
    f_missing: ComplexArray

    @property
    def dq_rad_m(self) -> float:
        return float(self.q_full_rad_m[1] - self.q_full_rad_m[0])


def _given_wavenumber_span(center_wavelength_nm: float, bandwidth_nm: float) -> float:
    """Convert a wavelength FWHM to the corresponding exact k-span."""
    lambda_center = center_wavelength_nm * 1e-9
    half_band = 0.5 * bandwidth_nm * 1e-9
    lambda_short = lambda_center - half_band
    lambda_long = lambda_center + half_band
    if lambda_short <= 0:
        raise ValueError("The wavelength band must remain positive.")
    k_high = 2.0 * np.pi / lambda_short
    k_low = 2.0 * np.pi / lambda_long
    return float(k_high - k_low)


def build_grids(config: SimulationConfig) -> GridSet:
    config.validate()

    given_span = _given_wavenumber_span(
        config.center_wavelength_nm,
        config.given_bandwidth_nm,
    )
    dq = given_span / config.n_given

    q_full = (np.arange(config.n_total, dtype=np.float64) - config.n_total / 2) * dq

    start = (config.n_total - config.n_given) // 2
    stop = start + config.n_given
    given_indices = np.arange(start, stop, dtype=np.int64)
    missing_indices = np.concatenate(
        [
            np.arange(0, start, dtype=np.int64),
            np.arange(stop, config.n_total, dtype=np.int64),
        ]
    )

    depth_dictionary = np.linspace(
        config.depth_min_um * 1e-6,
        config.depth_max_um * 1e-6,
        config.n_depth,
        endpoint=False,
        dtype=np.float64,
    )

    # Zero-padding in k gives a denser z sampling without changing the physical resolution.
    depth_reconstruction = np.arange(config.n_recon, dtype=np.float64) * (
        np.pi / (config.n_recon * dq)
    )

    q_given = q_full[given_indices]
    q_missing = q_full[missing_indices]

    f_given = np.exp(-1j * 2.0 * q_given[:, None] * depth_dictionary[None, :])
    f_missing = np.exp(-1j * 2.0 * q_missing[:, None] * depth_dictionary[None, :])

    return GridSet(
        carrier_wavenumber_rad_m=2.0 * np.pi / (config.center_wavelength_nm * 1e-9),
        q_full_rad_m=q_full,
        q_given_rad_m=q_given,
        q_missing_rad_m=q_missing,
        given_indices=given_indices,
        missing_indices=missing_indices,
        depth_dictionary_m=depth_dictionary,
        depth_reconstruction_m=depth_reconstruction,
        f_given=np.asarray(f_given, dtype=np.complex128),
        f_missing=np.asarray(f_missing, dtype=np.complex128),
    )
