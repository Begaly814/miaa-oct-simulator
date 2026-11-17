"""Synthetic OCT fringe simulator with IAA and MIAA reconstruction."""

from .config import SimulationConfig, load_config
from .grids import GridSet, build_grids
from .fringe import synthesize_two_reflector_fringe, make_wedge_depths
from .noise import add_complex_awgn
from .iaa import IAAResult, iaa_reconstruct
from .miaa import MIAAResult, miaa_extrapolate

__all__ = [
    "SimulationConfig",
    "load_config",
    "GridSet",
    "build_grids",
    "synthesize_two_reflector_fringe",
    "make_wedge_depths",
    "add_complex_awgn",
    "IAAResult",
    "iaa_reconstruct",
    "MIAAResult",
    "miaa_extrapolate",
]
