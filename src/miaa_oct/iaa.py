from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IAAResult:
    reflectivity: ComplexArray
    covariance: ComplexArray
    iterations: int
    relative_change_history: FloatArray


def _build_covariance(
    f_given: ComplexArray,
    reflectivity: ComplexArray,
    noise_variance: float,
    diagonal_loading: float,
) -> ComplexArray:
    power = np.abs(reflectivity) ** 2
    covariance = (f_given * power[None, :]) @ f_given.conj().T
    scale = float(np.trace(covariance).real / covariance.shape[0])
    regularizer = noise_variance + diagonal_loading * max(scale, 1.0)
    covariance = covariance + regularizer * np.eye(
        covariance.shape[0], dtype=np.complex128
    )
    # Remove tiny non-Hermitian roundoff before Cholesky factorization.
    return np.asarray(0.5 * (covariance + covariance.conj().T), dtype=np.complex128)


def iaa_reconstruct(
    y_given: ComplexArray,
    f_given: ComplexArray,
    noise_variance: float,
    iterations: int = 10,
    diagonal_loading: float = 1e-9,
    initial_reflectivity: ComplexArray | None = None,
    initial_covariance: ComplexArray | None = None,
    convergence_tolerance: float | None = None,
) -> IAAResult:
    """Brute-force IAA using one shared covariance inverse per iteration.

    The synthetic noise variance is supplied explicitly. This keeps the SNR experiment
    controlled and separates reconstruction error from noise-floor estimation error.
    """
    y = np.asarray(y_given, dtype=np.complex128)
    f = np.asarray(f_given, dtype=np.complex128)
    if y.ndim != 1 or f.ndim != 2 or f.shape[0] != y.size:
        raise ValueError("Expected y_given with shape (N,) and f_given with shape (N, L).")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if noise_variance < 0:
        raise ValueError("noise_variance must be nonnegative.")

    n_samples, n_depth = f.shape
    if initial_reflectivity is None:
        reflectivity = (f.conj().T @ y) / n_samples
    else:
        reflectivity = np.asarray(initial_reflectivity, dtype=np.complex128).copy()
        if reflectivity.shape != (n_depth,):
            raise ValueError("initial_reflectivity has the wrong shape.")

    covariance = None
    if initial_covariance is not None:
        covariance = np.asarray(initial_covariance, dtype=np.complex128).copy()
        if covariance.shape != (n_samples, n_samples):
            raise ValueError("initial_covariance has the wrong shape.")

    history: list[float] = []
    completed = 0
    for iteration in range(iterations):
        if covariance is None or iteration > 0:
            covariance = _build_covariance(
                f,
                reflectivity,
                noise_variance,
                diagonal_loading,
            )

        factor = cho_factor(covariance, lower=True, check_finite=False)
        r_inv_y = cho_solve(factor, y, check_finite=False)
        r_inv_f = cho_solve(factor, f, check_finite=False)

        numerator = f.conj().T @ r_inv_y
        denominator = np.sum(f.conj() * r_inv_f, axis=0)
        denominator_real = np.maximum(denominator.real, np.finfo(float).eps)
        updated = numerator / denominator_real

        relative_change = float(
            np.linalg.norm(updated - reflectivity)
            / max(np.linalg.norm(reflectivity), np.finfo(float).eps)
        )
        history.append(relative_change)
        reflectivity = np.asarray(updated, dtype=np.complex128)
        completed = iteration + 1

        if convergence_tolerance is not None and relative_change < convergence_tolerance:
            break

    final_covariance = _build_covariance(
        f,
        reflectivity,
        noise_variance,
        diagonal_loading,
    )
    return IAAResult(
        reflectivity=reflectivity,
        covariance=final_covariance,
        iterations=completed,
        relative_change_history=np.asarray(history, dtype=np.float64),
    )
