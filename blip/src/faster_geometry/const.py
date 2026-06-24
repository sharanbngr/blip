"""Constants."""

import math
from lisaconstants import SPEED_OF_LIGHT as CLIGHT, ASTRONOMICAL_YEAR as YEAR

__all__ = [
    "ARMLENGTH",
    "FSTAR",
    "CLIGHT",
    "YEAR",
    "INTERPOLATION_ALLOWED",
    "LOG_PERFORMANCE",
]

ARMLENGTH = 2.5e9
FSTAR = CLIGHT / (2 * math.pi * ARMLENGTH)

# Configuration of faster_geometry
INTERPOLATION_ALLOWED = True
LOG_PERFORMANCE = False
