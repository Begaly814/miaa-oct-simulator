from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    center_wavelength_nm: float = 510.0
    given_bandwidth_nm: float = 6.5
    n_total: int = 256
    n_given: int = 64
    n_recon: int = 4096

    depth_min_um: float = 300.0
    depth_max_um: float = 500.0
    n_depth: int = 512
    center_depth_um: float = 400.0

    num_a_lines: int = 256
    maximum_separation_um: float = 29.0
    minimum_separation_um: float = 0.0
    reflector_amplitudes: tuple[float, float] = (1.0, 1.0)

    cold_iterations: int = 8
    warm_iterations: int = 2
    diagonal_loading: float = 1e-9

    snr_levels_db: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0)
    monte_carlo_seeds: int = 10
    random_seed: int = 2026

    resolution_valley_fraction: float = 0.5
    peak_search_window_um: float = 12.0
    localization_tolerance_um: float = 5.0

    use_warm_start: bool = True
    edge_taper_fraction: float = 0.08

    @property
    def extrapolation_factor(self) -> float:
        return self.n_total / self.n_given

    def validate(self) -> None:
        if self.n_total <= self.n_given:
            raise ValueError("n_total must be larger than n_given for MIAA extrapolation.")
        if self.n_total % 2 or self.n_given % 2:
            raise ValueError("n_total and n_given must be even.")
        if self.n_recon < self.n_total:
            raise ValueError("n_recon must be at least n_total.")
        if self.n_recon % 2:
            raise ValueError("n_recon must be even.")
        if not 0 < self.edge_taper_fraction < 0.5:
            raise ValueError("edge_taper_fraction must lie between 0 and 0.5.")
        if self.depth_max_um <= self.depth_min_um:
            raise ValueError("depth_max_um must exceed depth_min_um.")


def _tuple_float(value: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in value)


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "reflector_amplitudes" in raw:
        raw["reflector_amplitudes"] = tuple(float(x) for x in raw["reflector_amplitudes"])
    if "snr_levels_db" in raw:
        raw["snr_levels_db"] = _tuple_float(raw["snr_levels_db"])

    config = SimulationConfig(**raw)
    config.validate()
    return config
