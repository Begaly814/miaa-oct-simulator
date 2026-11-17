from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def normalized_db(profile: ComplexArray, floor_db: float = -50.0) -> FloatArray:
    intensity = np.abs(np.asarray(profile, dtype=np.complex128)) ** 2
    peak = float(np.max(intensity))
    if peak <= 0:
        return np.full_like(intensity, floor_db, dtype=np.float64)
    values = 10.0 * np.log10(np.maximum(intensity / peak, 10.0 ** (floor_db / 10.0)))
    return np.asarray(values, dtype=np.float64)


def save_profiles_with_axes(
    output_path: str | Path,
    profiles: dict[str, tuple[FloatArray, ComplexArray]],
    true_depths_m: tuple[float, ...],
    x_limits_um: tuple[float, float] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for label, (depth_m, profile) in profiles.items():
        plt.plot(np.asarray(depth_m) * 1e6, normalized_db(profile), label=label)
    for z in true_depths_m:
        plt.axvline(z * 1e6, linestyle="--", linewidth=1)
    if x_limits_um is not None:
        plt.xlim(*x_limits_um)
    plt.ylim(-50, 1)
    plt.xlabel("Depth (µm)")
    plt.ylabel("Normalized intensity (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_a_line_comparison(
    output_path: str | Path,
    depth_m: FloatArray,
    profiles: dict[str, ComplexArray],
    true_depths_m: tuple[float, ...],
    x_limits_um: tuple[float, float] | None = None,
) -> None:
    save_profiles_with_axes(
        output_path,
        {name: (depth_m, profile) for name, profile in profiles.items()},
        true_depths_m,
        x_limits_um,
    )


def save_bscan(
    output_path: str | Path,
    bscan_complex: ComplexArray,
    depth_m: FloatArray,
    depth_limits_um: tuple[float, float],
    title: str,
    floor_db: float = -45.0,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    depth_um = np.asarray(depth_m) * 1e6
    mask = (depth_um >= depth_limits_um[0]) & (depth_um <= depth_limits_um[1])
    image = np.abs(np.asarray(bscan_complex)[:, mask]) ** 2
    image /= max(float(np.max(image)), np.finfo(float).eps)
    image_db = 10.0 * np.log10(np.maximum(image, 10.0 ** (floor_db / 10.0)))

    plt.figure(figsize=(9, 5))
    plt.imshow(
        image_db.T,
        origin="lower",
        aspect="auto",
        extent=[0, image_db.shape[0] - 1, depth_um[mask][0], depth_um[mask][-1]],
        vmin=floor_db,
        vmax=0,
        cmap="magma",
    )
    plt.colorbar(label="Normalized intensity (dB)")
    plt.xlabel("A-line index")
    plt.ylabel("Depth (µm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
