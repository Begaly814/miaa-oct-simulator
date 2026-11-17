from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


def add_complex_awgn(
    clean_signal: ComplexArray,
    snr_db: float,
    rng: np.random.Generator,
    exact_realization_snr: bool = True,
) -> tuple[ComplexArray, ComplexArray, float, float]:
    """Add circular complex Gaussian noise at a controlled power SNR.

    Returns noisy signal, noise, complex-sample noise variance, and measured SNR in dB.
    """
    clean = np.asarray(clean_signal, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(clean) ** 2))
    if signal_power <= 0:
        raise ValueError("clean_signal must contain nonzero power.")

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = (
        rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    ) / np.sqrt(2.0)

    if exact_realization_snr:
        realized_unit_power = float(np.mean(np.abs(noise) ** 2))
        noise *= np.sqrt(target_noise_power / realized_unit_power)
    else:
        noise *= np.sqrt(target_noise_power)

    noisy = clean + noise
    measured_noise_power = float(np.mean(np.abs(noise) ** 2))
    measured_snr_db = 10.0 * np.log10(signal_power / measured_noise_power)
    return (
        np.asarray(noisy, dtype=np.complex128),
        np.asarray(noise, dtype=np.complex128),
        measured_noise_power,
        float(measured_snr_db),
    )
