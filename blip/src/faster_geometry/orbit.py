"""Orbital calculations."""

import chex
from jax import numpy as jnp

from astropy import units
from astropy.coordinates import SkyCoord

try:
    import lisaorbits
    has_lisaorbits = True
except ModuleNotFoundError:
    has_lisaorbits = False
    pass

from .const import ARMLENGTH

LINKS = [12, 23, 31, 13, 32, 21]

__all__ = [
    "compute_orbits",
    "get_arm_orientations",
    "get_sc_positions",
    "get_link_vectors",
]


def compute_orbits(times, use_lisaorbits=False, betaphase=0):
    """
    Compute orbit information at specified time array.

    Parameters
    ----------
    times : array 1D
        Times at which the positions of the spacecraft and link vectors should be
        computed.

    Returns
    -------
    array 1D
        the input times array
    array (3, ntimes, 3)
        Spacecraft positions in ecliptic cartesian coordinates as an array, where the
        first dimension specifies the spacecraft.
    array (6, ntimes, 3)
        Single link unit vectors in lisaorbits order.
    """
    chex.assert_rank(times, 1)

    if use_lisaorbits:
        return _orbits_from_lisaorbits(times)

    ## Semimajor axis in m
    a = 1.496e11

    ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
    alphaphase = 0

    ## Orbital angle alpha(t)
    at = (2 * jnp.pi / 31557600) * times + alphaphase

    ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
    e = ARMLENGTH / (2 * a * jnp.sqrt(3))

    ## Initialize arrays
    beta_n = (2 / 3) * jnp.pi * jnp.array([0, 1, 2]) + betaphase

    ## meshgrid arrays
    Beta_n, Alpha_t = jnp.meshgrid(beta_n, at)

    ## Calculate inclination and positions for each satellite.
    x_n = a * jnp.cos(Alpha_t) + a * e * (
        jnp.sin(Alpha_t) * jnp.cos(Alpha_t) * jnp.sin(Beta_n)
        - (1 + (jnp.sin(Alpha_t)) ** 2) * jnp.cos(Beta_n)
    )
    y_n = a * jnp.sin(Alpha_t) + a * e * (
        jnp.sin(Alpha_t) * jnp.cos(Alpha_t) * jnp.cos(Beta_n)
        - (1 + (jnp.cos(Alpha_t)) ** 2) * jnp.sin(Beta_n)
    )
    z_n = -jnp.sqrt(3) * a * e * jnp.cos(Alpha_t - Beta_n)

    ## Construct position vectors r_n
    rs1 = jnp.array([x_n[:, 0], y_n[:, 0], z_n[:, 0]])
    rs2 = jnp.array([x_n[:, 1], y_n[:, 1], z_n[:, 1]])
    rs3 = jnp.array([x_n[:, 2], y_n[:, 2], z_n[:, 2]])

    chex.assert_shape([rs1, rs2, rs3], (3, times.shape[0]))
    sc_positions = jnp.stack([rs1.T, rs2.T, rs3.T])

    lv12 = rs1 - rs2
    lv23 = rs2 - rs3
    lv31 = rs3 - rs1
    lv12 = lv12 / jnp.linalg.norm(lv12, axis=0)[jnp.newaxis, :]
    lv23 = lv23 / jnp.linalg.norm(lv23, axis=0)[jnp.newaxis, :]
    lv31 = lv31 / jnp.linalg.norm(lv31, axis=0)[jnp.newaxis, :]
    lv13 = -lv31
    lv32 = -lv23
    lv21 = -lv12
    link_vectors = jnp.stack([lv12.T, lv23.T, lv31.T, lv13.T, lv32.T, lv21.T])

    chex.assert_shape(sc_positions, (3, times.shape[0], 3))
    chex.assert_shape(link_vectors, (6, times.shape[0], 3))

    return (times, sc_positions, link_vectors)


def _orbits_from_lisaorbits(times):
    """
    Compute orbit information at specified time array.

    Parameters
    ----------
    times : array 1D
        Times at which the positions of the spacecraft and link vectors should be
        computed.

    Returns
    -------
    array 1D
        the input times array
    array (3, ntimes, 3)
        Spacecraft positions in ecliptic cartesian coordinates as an array, where the
        first dimension specifies the spacecraft.
    array (6, ntimes, 3)
        Single link unit vectors in lisaorbits order.
    """
    assert has_lisaorbits, "lisaorbits is not installed"
    chex.assert_rank(times, 1)

    # We are interfacing with non-jax-traceable numpy code:
    # therefore, compute all that we need from it upfront.
    orbits = lisaorbits.EqualArmlengthOrbits()
    sc_positions_icrs = jnp.array(
        orbits.compute_position(times, [1, 2, 3]).transpose(1, 0, 2)
    )
    link_vectors_icrs = jnp.array(orbits.compute_unit_vector(times).transpose(1, 0, 2))
    chex.assert_shape(sc_positions_icrs, (3, times.shape[0], 3))
    chex.assert_shape(link_vectors_icrs, (6, times.shape[0], 3))

    # v3 means ICRS equatorial coordinates. v2 => ecliptic.
    assert lisaorbits.__version__ >= "3"

    sc_positions = []
    link_vectors = []
    for i in range(3):
        sc_positions.append(_icrs_to_ecliptic(sc_positions_icrs[i]))
    for i in range(6):
        link_vectors.append(_icrs_to_ecliptic(link_vectors_icrs[i]))

    sc_positions = jnp.asarray(sc_positions)
    link_vectors = jnp.asarray(link_vectors)
    chex.assert_shape(sc_positions, (3, times.shape[0], 3))
    chex.assert_shape(link_vectors, (6, times.shape[0], 3))

    return (times, sc_positions, link_vectors)


def get_arm_orientations(t, sc, orbits):
    """
    Get unit vectors for left and right arm of a given spacecraft.

    The numbering convention is the same as in lisaorbits.

    Parameters
    ----------
    t : float
        time
    sc : int
        Spacecraft number (1, 2, 3). Must be known at JAX trace-time.
    orbits : tuple
        orbital information from compute_orbits().

    Returns
    -------
    array (3,)
        Unit vector pointing away from the given spacecraft (1 -> 3, 2 -> 1, 3 -> 2).
    array (3,)
        Unit vector pointing away from the given spacecraft (1 -> 2, 2 -> 3, 3 -> 1).
    """
    # sc=1,2,3 must be trace-time known
    chex.assert_shape([t, sc], ())
    assert 1 <= sc and sc <= 3

    # The lisaconstants indexing convention for links is receiver-sender.
    # In our case we want the directions of the links pointing away from
    # a given spacecraft.
    if sc == 1:
        idx_u, idx_v = 5, 2
        assert LINKS[idx_u] == 21
        assert LINKS[idx_v] == 31
    elif sc == 2:
        idx_u, idx_v = 4, 0
        assert LINKS[idx_u] == 32
        assert LINKS[idx_v] == 12
    else:
        idx_u, idx_v = 3, 1
        assert LINKS[idx_u] == 13
        assert LINKS[idx_v] == 23

    lv = get_link_vectors(t, orbits)
    chex.assert_shape(lv, (6, 3))
    u = lv[idx_u]
    v = lv[idx_v]

    chex.assert_shape([u, v], (3,))
    return u, v


# lisaorbits v3 needs equatorial ICRS frame, but in BLIP we use ecliptic always
def _icrs_to_ecliptic(positions_icrs):
    chex.assert_shape(positions_icrs, (None, 3))
    ntimes = positions_icrs.shape[0]

    x = positions_icrs[:, 0]
    y = positions_icrs[:, 1]
    z = positions_icrs[:, 2]
    coords = SkyCoord(
        x=x, y=y, z=z, unit="m", representation_type="cartesian", frame="icrs"
    )
    coords = coords.transform_to("barycentricmeanecliptic")
    coords.representation_type = "cartesian"
    x = jnp.array(coords.x / units.meter)  # pylint: disable=no-member
    y = jnp.array(coords.y / units.meter)  # pylint: disable=no-member
    z = jnp.array(coords.z / units.meter)  # pylint: disable=no-member

    positions_eclp = jnp.stack([x, y, z]).T
    chex.assert_shape(positions_eclp, (ntimes, 3))
    return positions_eclp


# The following traceable jax functions just look up values from the pre-computed arrays
# sc_positions and link_vectors.
def get_sc_positions(t, orbits):
    """
    Look up S/C positions in orbits, in a way that is jax-traceable.

    This function does not perform interpolation. It will return the 'last known'
    position of the spacecraft.

    Parameters
    ----------
    t : float
        time.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    array (3, 3)
        The positions of the three spacecraft (first axis) in cartesian coordinates
        (second axis).
    """
    chex.assert_shape(t, ())
    times, sc_positions, _ = orbits
    chex.assert_rank(times, 1)
    chex.assert_shape(sc_positions, (3, times.shape[0], 3))

    idx = jnp.searchsorted(times, t)
    res = sc_positions[:, idx, :]
    chex.assert_shape(res, (3, 3))
    return res


def get_link_vectors(t, orbits):
    """
    Look up link unit vectors in orbits, in a way that is jax-traceable.

    This function does not perform interpolation. It will return the 'last known' unit
    vectors.

    Parameters
    ----------
    t : float
        time.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    array (6, 3)
        The unit vectors in 3D (second axis) for each of the single links (first axis)
        in lisaorbits order.
    """
    chex.assert_shape(t, ())
    times, _, link_vectors = orbits
    chex.assert_rank(times, 1)
    chex.assert_shape(link_vectors, (6, times.shape[0], 3))
    idx = jnp.searchsorted(times, t)
    res = link_vectors[:, idx, :]
    chex.assert_shape(res, (6, 3))
    return res
