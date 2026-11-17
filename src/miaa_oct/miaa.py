from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class MIAAResult:
    full_spectrum: ComplexArray
    missing_spectrum: ComplexArray
    missing_indices: NDArray[np.int64]


def miaa_extrapolate(
    y_given: ComplexArray,
    reflectivity: ComplexArray,
    covariance_given: ComplexArray,
    f_given: ComplexArray,
    f_missing: ComplexArray,
    given_indices: NDArray[np.int64],
    missing_indices: NDArray[np.int64],
    n_total: int,
) -> MIAAResult:
    """Estimate missing spectral samples using the MIAA MMSE formula.

    Implements y_m = sum_l [|a_l|^2 f_g,l^H R_g^-1 y_g] f_m,l.
    The measured samples are copied unchanged into the returned full spectrum.
    """
    y = np.asarray(y_given, dtype=np.complex128)
    a = np.asarray(reflectivity, dtype=np.complex128)
    r = np.asarray(covariance_given, dtype=np.complex128)
    fg = np.asarray(f_given, dtype=np.complex128)
    fm = np.asarray(f_missing, dtype=np.complex128)

    factor = cho_factor(r, lower=True, check_finite=False)
    r_inv_y = cho_solve(factor, y, check_finite=False)
    cross_projection = fg.conj().T @ r_inv_y
    coefficients = (np.abs(a) ** 2) * cross_projection
    missing = fm @ coefficients

    full = np.zeros(n_total, dtype=np.complex128)
    full[np.asarray(given_indices, dtype=np.int64)] = y
    full[np.asarray(missing_indices, dtype=np.int64)] = missing

    return MIAAResult(
        full_spectrum=full,
        missing_spectrum=np.asarray(missing, dtype=np.complex128),
        missing_indices=np.asarray(missing_indices, dtype=np.int64),
    )
