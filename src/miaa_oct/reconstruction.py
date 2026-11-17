from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def centered_zero_pad_spectrum(
    spectrum: ComplexArray,
    output_length: int,
) -> ComplexArray:
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    if output_length < spectrum.size:
        raise ValueError("output_length cannot be smaller than the input spectrum.")
    if (output_length - spectrum.size) % 2:
        raise ValueError("The padding difference must be even to preserve centering.")
    padded = np.zeros(output_length, dtype=np.complex128)
    start = (output_length - spectrum.size) // 2
    padded[start : start + spectrum.size] = spectrum
    return padded


def reconstruct_from_uniform_q(
    spectrum_centered: ComplexArray,
    dq_rad_m: float,
    n_recon: int,
) -> tuple[FloatArray, ComplexArray]:
    """Inverse transform a centered, uniformly q-sampled analytic fringe."""
    padded = centered_zero_pad_spectrum(spectrum_centered, n_recon)
    image = np.fft.ifft(np.fft.ifftshift(padded))
    depth = np.arange(n_recon, dtype=np.float64) * np.pi / (n_recon * dq_rad_m)
    return depth, np.asarray(image, dtype=np.complex128)


def given_band_dft_spectrum(
    y_given: ComplexArray,
    given_indices: NDArray[np.int64],
    n_total: int,
) -> ComplexArray:
    full = np.zeros(n_total, dtype=np.complex128)
    full[np.asarray(given_indices, dtype=np.int64)] = np.asarray(
        y_given, dtype=np.complex128
    )
    return full


def squared_cosine_edge_taper(length: int, fraction: float) -> FloatArray:
    if length < 2:
        raise ValueError("length must be at least 2.")
    edge = max(1, int(round(length * fraction)))
    if 2 * edge >= length:
        raise ValueError("Taper edges are too long for the spectrum length.")
    window = np.ones(length, dtype=np.float64)
    phase = np.linspace(0.0, np.pi / 2.0, edge, endpoint=False)
    ramp = np.sin(phase) ** 2
    window[:edge] = ramp
    window[-edge:] = ramp[::-1]
    return window
