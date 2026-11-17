import numpy as np

from miaa_oct.config import SimulationConfig
from miaa_oct.fringe import synthesize_two_reflector_fringe
from miaa_oct.grids import build_grids
from miaa_oct.miaa import miaa_extrapolate
from miaa_oct.iaa import iaa_reconstruct
from miaa_oct.noise import add_complex_awgn


def small_config() -> SimulationConfig:
    return SimulationConfig(
        n_total=64,
        n_given=16,
        n_recon=512,
        n_depth=64,
        cold_iterations=3,
        num_a_lines=16,
    )


def test_grid_shapes() -> None:
    config = small_config()
    grids = build_grids(config)
    assert grids.f_given.shape == (config.n_given, config.n_depth)
    assert grids.f_missing.shape == (config.n_total - config.n_given, config.n_depth)
    assert np.intersect1d(grids.given_indices, grids.missing_indices).size == 0


def test_exact_snr_scaling() -> None:
    rng = np.random.default_rng(1)
    clean = np.ones(32, dtype=np.complex128)
    _, _, _, measured = add_complex_awgn(clean, 25.0, rng)
    assert abs(measured - 25.0) < 1e-10


def test_miaa_preserves_measured_samples() -> None:
    config = small_config()
    grids = build_grids(config)
    clean = synthesize_two_reflector_fringe(
        grids.q_full_rad_m,
        (390e-6, 410e-6),
    )
    y = clean[grids.given_indices]
    result = iaa_reconstruct(
        y,
        grids.f_given,
        noise_variance=1e-8,
        iterations=2,
    )
    miaa = miaa_extrapolate(
        y,
        result.reflectivity,
        result.covariance,
        grids.f_given,
        grids.f_missing,
        grids.given_indices,
        grids.missing_indices,
        config.n_total,
    )
    np.testing.assert_allclose(miaa.full_spectrum[grids.given_indices], y)
