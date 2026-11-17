from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def synthesize_reflector_fringe(
    q_rad_m: FloatArray,
    depths_m: Sequence[float],
    amplitudes: Sequence[complex],
    carrier_wavenumber_rad_m: float | None = None,
) -> ComplexArray:
    """Synthesize a normalized analytic OCT fringe from discrete reflectors.

    q is the centered wavenumber coordinate q = k - k_c. If k_c is supplied,
    the physical round-trip carrier phase exp(-i 2 k_c z_j) is absorbed into
    each complex reflector amplitude.
    """
    q = np.asarray(q_rad_m, dtype=np.float64)
    z = np.asarray(depths_m, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.complex128)
    if z.ndim != 1 or a.ndim != 1 or z.size != a.size or z.size == 0:
        raise ValueError("depths_m and amplitudes must be nonempty vectors of equal length.")

    if carrier_wavenumber_rad_m is not None:
        a = a * np.exp(-1j * 2.0 * float(carrier_wavenumber_rad_m) * z)

    phase = np.exp(-1j * 2.0 * q[:, None] * z[None, :])
    return np.asarray(phase @ a, dtype=np.complex128)


def synthesize_two_reflector_fringe(
    q_rad_m: FloatArray,
    depths_m: tuple[float, float],
    amplitudes: tuple[complex, complex] = (1.0 + 0j, 1.0 + 0j),
    carrier_wavenumber_rad_m: float | None = None,
) -> ComplexArray:
    return synthesize_reflector_fringe(
        q_rad_m,
        depths_m,
        amplitudes,
        carrier_wavenumber_rad_m=carrier_wavenumber_rad_m,
    )


def make_wedge_depths(
    num_a_lines: int,
    center_depth_m: float,
    maximum_separation_m: float,
    minimum_separation_m: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Create two reflectors that approach each other across the B-scan."""
    if num_a_lines < 2:
        raise ValueError("num_a_lines must be at least 2.")
    if maximum_separation_m < minimum_separation_m:
        raise ValueError("maximum_separation_m must be >= minimum_separation_m.")

    separation = np.linspace(
        maximum_separation_m,
        minimum_separation_m,
        num_a_lines,
        dtype=np.float64,
    )
    z1 = center_depth_m - separation / 2.0
    z2 = center_depth_m + separation / 2.0
    return z1, z2, separation
