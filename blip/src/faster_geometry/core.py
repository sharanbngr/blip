"""
Core logic for LISA response calculations.

Here is the logic, top-down.

- :func:`mich_response_unconvolved` computes the response correlation matrix. This is
  the ultimate goal.
- It calls :func:`mich_antenna_pattern` which computes the LISA response for individual
  TDI spacecraft or TDI channels, and individual GW polarization modes.
- :func:`mich_antenna_pattern` simply multiplies a detector tensor by a polarization
  tensor.
- To compute the detector tensor you need orbital information as well as the
  :func:`timing transfer function <timing_transfer_fn>`.
- To define a polarization tensor you need an orthonormal basis at the tangent space to
  each spacecraft (:func:`get_ortho_basis_ecliptic_3d`).
"""

########### Implementation details ##########
# This module leans heavily on JAX automatic vectorization and JIT compilation. All the
# functions are written for the simplest possible array shapes (mostly scalars), making
# them easier to check for correctness. Trace-time assertions, mostly on array shapes,
# have also been placed as strong comments all throughout the code.

# The formulas implemented here are exactly the ones in the original BLIP paper
# Banagiri+21, with two exceptions:
# - the mistaken factor of 1/4pi is removed from eq. (18) which defines the response
#   matrix;
# - all throughout this module, the sign of n is flipped wrt. the paper, i.e. the vector
#   n here means the direction towards the GW source, not from the source.

# TODO check if the sign of n conflicts with the rest of BLIP.


from jax import numpy as jnp
import chex

from .orbit import get_arm_orientations, get_sc_positions
from .const import FSTAR, CLIGHT

__all__ = [
    "mich_response_unconvolved",
    "mich_antenna_pattern",
    "mich_detector_tensor",
    "get_ortho_basis_ecliptic_3d",
    "timing_transfer_fn",
]


def mich_response_unconvolved(t, f, n, orbits):
    r"""
    Unconvolved Michelson (TDI gen 0) sky SGWB response.

    Parameters
    ----------
    t : float
        time
    f : float
        frequency
    n : array (3,)
        normalized vector in the direction of the GW source.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    complex array (3, 3)
        Unconvolved response matrix for the three data channels.

    Notes
    -----
    The quantity computed here is exactly

    .. math::

        \frac{1}{2} \sum_{A=+,\times}\left(F_I^A(f, \mathbf{n})^* F_J^A(f, \mathbf{n})
        \right)

    where :math:`I` and :math:`J` stand for TDI channels, and :math:`F_I^A` are antenna
    pattern functions.

    This is integrated against the sky map to produce the GW time-frequency correlation
    matrix.

    The convention here agrees with Criswell+25 eq. (6) (but for a complex conjugate),
    which is a corrected version of Banagiri+21 eq. (18).
    """
    chex.assert_shape([t, f], ())
    chex.assert_shape(n, (3,))

    res = jnp.zeros((3, 3), dtype=complex)

    # This loop intentionally uses python control flow so that it is
    # unrolled in tracing and the channels (c1, c2) are trace-time known.
    for c1 in range(3):
        for c2 in range(c1, 3):
            fp1 = mich_antenna_pattern(t, f, n, "plus", c1, orbits)
            fp2 = mich_antenna_pattern(t, f, n, "plus", c2, orbits)
            fc1 = mich_antenna_pattern(t, f, n, "cross", c1, orbits)
            fc2 = mich_antenna_pattern(t, f, n, "cross", c2, orbits)
            chex.assert_shape([fp1, fp2, fc1, fc2], ())
            res = res.at[c1, c2].set(0.5 * (fp1.conj() * fp2 + fc1.conj() * fc2))
            if c1 != c2:
                res = res.at[c2, c1].set(res[c1, c2].conj())

    chex.assert_shape(res, (3, 3))
    return res


def mich_antenna_pattern(t, f, n, polarization: str, channel, orbits):
    """
    Compute Michelson (TDI gen 0) antenna pattern.

    Checked against Banagiri+21 and Romano & Cornish 2017.

    Parameters
    ----------
    t : float
        time
    f : float
        frequency
    n : array (3,)
        Unit vector in the direction of the GW source.
    polarization : str
        Should be "plus" or "cross". Must be known at JAX trace time.
    channel : int
        Channel index 0, 1, 2. Must be known at JAX trace time.
    orbits : tuple
        orbital information returned by compute_orbits().

    Returns
    -------
    complex
        The antenna pattern.
    """
    # polarization and channel must be trace-time known
    chex.assert_shape([t, f, channel], ())
    chex.assert_shape(n, (3,))
    assert polarization in ["plus", "cross"]
    assert 0 <= channel and channel < 3

    # lam, beta = ecliptic coordinates (lon, lat) of n
    n = n / jnp.linalg.norm(n)
    beta = jnp.arcsin(n[2])
    lam = jnp.atan2(n[1], n[0])
    lam = jnp.where(lam < 0, lam + 2 * jnp.pi, lam)

    # polarization tensor, Romano & Cornish 2017 eq 2.3
    _, ell, emm = get_ortho_basis_ecliptic_3d(lam, beta)
    if polarization == "plus":
        pol_tens = jnp.outer(ell, ell) - jnp.outer(emm, emm)
    else:
        pol_tens = jnp.outer(ell, emm) + jnp.outer(emm, ell)

    # detector tensor
    sc = channel + 1
    u, v = get_arm_orientations(t, sc, orbits)
    r = get_sc_positions(t, orbits)[sc - 1]
    det_tens = mich_detector_tensor(f, u, v, n, r)

    chex.assert_shape([det_tens, pol_tens], (3, 3))
    res = jnp.tensordot(det_tens, pol_tens)
    chex.assert_shape(res, ())
    return res


def mich_detector_tensor(f, u, v, n, r):
    """
    Michelson channel detector tensor.

    Checked against Banagiri+21 eq (15).

    Parameters
    ----------
    f : float
        frequency in Hz
    u : array (3,)
        normalized vector in the direction of the first arm
    v : array (3,)
        normalized vector in the direction of the second arm
    n : array (3,)
        normalized vector in the direction of the GW source
    r : array (3,)
        position of vertex S/C in barycentric ecliptic cartesian coordinates

    Returns
    -------
    complex array (3, 3)
        detector tensor
    """
    chex.assert_shape(f, ())
    chex.assert_shape([u, v, n, r], (3,))

    uu = jnp.outer(u, u)
    vv = jnp.outer(v, v)
    chex.assert_shape([uu, vv], (3, 3))
    un = jnp.dot(u, n)
    vn = jnp.dot(v, n)
    nr = jnp.dot(n, r)
    omega = 2 * jnp.pi * f
    chex.assert_shape([un, vn, nr, omega], ())

    tun = timing_transfer_fn(f, un)
    tvn = timing_transfer_fn(f, vn)
    chex.assert_shape([tun, tvn], ())

    factor = jnp.exp(1j * omega * nr / CLIGHT)
    result = 0.5 * factor * (tun * uu - tvn * vv)

    chex.assert_shape(result, (3, 3))
    return result


def timing_transfer_fn(f, costheta):
    """
    Timing transfer function for two-way photon propagation.

    Checked against Banagiri+21 eq (16) and Cornish & Rubbo 2003 eq (37). Also agrees
    with Romano & Cornish 2017 eq (5.27) up to a constant 2L/c. This seems due to the
    conversion between strain and timing measurements, eq (5.4) in the living review.

    Parameters
    ----------
    f : float
        frequency in Hz
    costheta : float
        cosine of angle between arm and sky direction

    Returns
    -------
    complex
        the transfer function.
    """
    chex.assert_shape([f, costheta], ())

    f0 = f / (2 * FSTAR)
    s1 = _sinc(f0 * (1 + costheta))
    s2 = _sinc(f0 * (1 - costheta))
    e1 = jnp.exp(-1j * f0 * (3 - costheta))
    e2 = jnp.exp(-1j * f0 * (1 - costheta))
    res = 0.5 * (s1 * e1 + s2 * e2)

    chex.assert_shape(res, ())
    return res


def get_ortho_basis_ecliptic_3d(lam, beta):
    """
    Get right-handed orthonormal basis (n, l, m).

    This is the basis in Romano & Cornish 2017 eq (2.4).

    Parameters
    ----------
    lam : float
        ecliptic longitude
    beta : float
        ecliptic latitude

    Returns
    -------
    tuple (n, l, m)
        A tuple of arrays of shape (3,).
    """
    chex.assert_shape([lam, beta], ())

    theta, phi = jnp.pi / 2 - beta, lam

    ct, st = jnp.cos(theta), jnp.sin(theta)
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    enn = jnp.array([st * cp, st * sp, ct])
    ell = jnp.array([ct * cp, ct * sp, -st])
    emm = jnp.array([-sp, cp, 0])

    chex.assert_shape([enn, ell, emm], (3,))
    return enn, ell, emm


# Surprisingly, this does not exist in jax.scipy.special
def _sinc(x):
    # Inner select avoids NaN when differentiating at x=0
    _x = jnp.select([x != 0, True], [x, 1.0])
    return jnp.select([x != 0, True], [jnp.sin(_x) / _x, 1.0])
