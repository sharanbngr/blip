"""Sparse grids and interpolation."""

import logging

import chex
from jax import numpy as jnp, vmap

from .const import YEAR

__all__ = [
    "get_sparse_tf_grid",
    "get_response_interpolator",
    "DT_SPARSE",
    "DF_SPARSE",
    "FMAX_SPARSE",
    "TOL_INTERP",
]

logger = logging.getLogger(__name__)


# Inside rectangular patches of sides (DT_SPARSE, DF_SPARSE), with frequencies no more
# than FMAX_SPARSE, we guarantee interpolation precision TOL_INTERP.
# See test_interpolation_response() in test_hyp_faster_geometry.py.
DT_SPARSE = 2 * 86400  # two days
DF_SPARSE = 1e-3  # 1 mHz
FMAX_SPARSE = 10e-3  # 10 mHz
TOL_INTERP = 2e-3


def get_sparse_tf_grid(times, freqs):
    """
    Generate a sparse time-frequency grid suitable for interpolation.

    Parameters
    ----------
    times : array (ntimes,)
        dense time grid, assumed to start in the beginning of the year
    freqs : array (nfreqs,)
        dense frequency grid

    Returns
    -------
    array 1D
        sparse time grid up to t = 1 year.
    array 1D
        sparse frequency grid
    """
    chex.assert_rank([times, freqs], 1)

    if len(times) == 1:
        times_sparse = times
    else:
        dt = times[1] - times[0]
        dt_s = DT_SPARSE
        if dt > dt_s:
            times_sparse = times
        else:
            tmax = jnp.minimum(times[-1], YEAR)
            times_sparse = jnp.arange(times[0], tmax, dt_s)

    if len(freqs) == 1:  # or freqs[-1] > FMAX_SPARSE:
        freqs_sparse = freqs
    else:
        df = freqs[1] - freqs[0]
        df_s = DF_SPARSE
        if df > df_s:
            freqs_sparse = freqs
        else:
            freqs_sparse = jnp.arange(freqs[0], freqs[-1], df_s)
            if freqs[-1] > FMAX_SPARSE:
                logger.warning(
                    f"max freq is {freqs[-1]:.2e} > {FMAX_SPARSE:.2e}"
                    f" but allowing interpolation anyway. Make sure your model spectrum"
                    f" is insignificant after f = {FMAX_SPARSE:.2e}"
                )

    chex.assert_rank([times_sparse, freqs_sparse], 1)
    return times_sparse, freqs_sparse


def get_response_interpolator(times, freqs, times_sparse, freqs_sparse, periodic=True):
    """
    Generate interpolator function for response matrices.

    The interpolated response assumes 1-year periodicity for time.

    Parameters
    ----------
    times : array (nt,)
        dense (target) time grid
    freqs : array (nf,)
        dense (target) frequency grid
    times_sparse : array (nt_s,)
        sparse time grid
    freqs_sparse : array (nf_s,)
        sparse frequency grid
    periodic : bool, optional
        If True, interpolate on times with 1-year period in accordance with
        :fun:`get_sparse_tf_grid`; by default True

    Returns
    -------
    function
        an interpolator that receives a complex array of shape (3, 3, nf_s, nt_s) (the
        sparse response matrix) and returns a complex array (3, 3, nf, nt).

    Examples
    --------
    >>> t_sparse, f_sparse = get_sparse_tf_grid(times, freqs)
    >>> _interpolate = get_response_interpolator(times, freqs, t_sparse, f_sparse)
    >>> response = jax.jit(_interpolate)(response_sparse)
    """
    chex.assert_rank([times, freqs, times_sparse, freqs_sparse], 1)
    nt, nf, nt_s, nf_s = len(times), len(freqs), len(times_sparse), len(freqs_sparse)

    period = YEAR if periodic else None
    interp_vect = vmap(lambda r: jnp.interp(times, times_sparse, r, period=period))
    interp_vecf = vmap(lambda r: jnp.interp(freqs, freqs_sparse, r))

    def interpolator(response_sparse):
        chex.assert_shape(response_sparse, (3, 3, nf_s, nt_s))

        # Global idea:
        #       interp time         interp freq
        # rs1 --------------> rs2 --------------> rs3.

        rs1 = response_sparse

        # rs1b = rs1, reshaped for interpolation along times
        rs1b = rs1.reshape((-1, nt_s))

        rs2b = interp_vect(rs1b)

        rs2 = rs2b.reshape((3, 3, nf_s, nt))

        # rs2c = rs2, reshaped for interpolation along frequencies
        rs2c = rs2.transpose(0, 1, 3, 2).reshape((-1, nf_s))

        rs3c = interp_vecf(rs2c)

        rs3 = rs3c.reshape((3, 3, nt, nf)).transpose(0, 1, 3, 2)

        chex.assert_shape(rs3, (3, 3, nf, nt))
        return rs3

    return interpolator
