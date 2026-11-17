from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ResolutionResult:
    resolved: bool
    estimated_depths_m: tuple[float, float] | None
    peak_intensities: tuple[float, float] | None
    valley_intensity: float | None
    localization_rmse_m: float | None
    amplitude_ratio_error_db: float | None


def normalized_mse_db(estimate: ComplexArray, truth: ComplexArray) -> float:
    estimate = np.asarray(estimate, dtype=np.complex128)
    truth = np.asarray(truth, dtype=np.complex128)
    denominator = float(np.sum(np.abs(truth) ** 2))
    if denominator <= 0:
        raise ValueError("truth must contain nonzero energy.")
    numerator = float(np.sum(np.abs(estimate - truth) ** 2))
    return float(10.0 * np.log10(max(numerator, np.finfo(float).tiny) / denominator))


def evaluate_two_reflector_resolution(
    depth_m: FloatArray,
    complex_profile: ComplexArray,
    true_depths_m: tuple[float, float],
    valley_fraction: float = 0.5,
    search_window_m: float = 12e-6,
    localization_tolerance_m: float = 5e-6,
) -> ResolutionResult:
    depth = np.asarray(depth_m, dtype=np.float64)
    intensity = np.abs(np.asarray(complex_profile, dtype=np.complex128)) ** 2
    z_true = np.sort(np.asarray(true_depths_m, dtype=np.float64))

    region_mask = (depth >= z_true[0] - search_window_m) & (
        depth <= z_true[1] + search_window_m
    )
    region_indices = np.flatnonzero(region_mask)
    if region_indices.size < 3:
        return ResolutionResult(False, None, None, None, None, None)

    local_intensity = intensity[region_indices]
    dz = float(np.median(np.diff(depth)))
    minimum_peak_distance = max(1, int(round(max(0.5e-6, 0.15 * (z_true[1] - z_true[0])) / dz)))
    peaks, _ = find_peaks(local_intensity, distance=minimum_peak_distance)
    if peaks.size < 2:
        return ResolutionResult(False, None, None, None, None, None)

    strongest = peaks[np.argsort(local_intensity[peaks])[-2:]]
    strongest = np.sort(strongest)
    global_peaks = region_indices[strongest]
    z_est = depth[global_peaks]
    p = intensity[global_peaks]

    between = intensity[global_peaks[0] : global_peaks[1] + 1]
    valley = float(np.min(between))
    localization_errors = z_est - z_true
    rmse = float(np.sqrt(np.mean(localization_errors**2)))
    amplitude_error_db = float(abs(10.0 * np.log10(max(p[0], 1e-300) / max(p[1], 1e-300))))

    valley_ok = valley <= valley_fraction * float(np.min(p))
    localization_ok = bool(np.all(np.abs(localization_errors) <= localization_tolerance_m))
    resolved = bool(valley_ok and localization_ok)

    return ResolutionResult(
        resolved=resolved,
        estimated_depths_m=(float(z_est[0]), float(z_est[1])),
        peak_intensities=(float(p[0]), float(p[1])),
        valley_intensity=valley,
        localization_rmse_m=rmse,
        amplitude_ratio_error_db=amplitude_error_db,
    )
