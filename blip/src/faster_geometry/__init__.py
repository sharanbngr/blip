"""
Functions to compute the LISA response to stochastic backgrounds.

The main procedure is :func:`calculate_response_functions`, which is used to interface
with the rest of BLIP.

The main objective is to compute the LISA response for a given sky direction, frequency
and time. This is done in :func:`mich_response_unconvolved`.
"""

import logging

from .interface import calculate_response_functions
from .core import (
    mich_response_unconvolved,
    mich_antenna_pattern,
    mich_detector_tensor,
    get_ortho_basis_ecliptic_3d,
    timing_transfer_fn,
)
from .const import ARMLENGTH, FSTAR, YEAR, CLIGHT
from .orbit import (
    compute_orbits,
    get_arm_orientations,
    get_link_vectors,
    get_sc_positions,
)
from .util import get_vecs_all_sky

__all__ = [
    "calculate_response_functions",
    "mich_response_unconvolved",
    "mich_antenna_pattern",
    "mich_detector_tensor",
    "get_ortho_basis_ecliptic_3d",
    "timing_transfer_fn",
    "ARMLENGTH",
    "FSTAR",
    "YEAR",
    "CLIGHT",
    "compute_orbits",
    "get_arm_orientations",
    "get_link_vectors",
    "get_sc_positions",
    "get_vecs_all_sky",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # TODO set this in run_blip, use config option
logger.addHandler(logging.StreamHandler())
