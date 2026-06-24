from jax import numpy as jnp
import chex
import healpy as hp

__all__ = ["get_vecs_all_sky"]


def get_vecs_all_sky(nside):
    """
    Compute array of all unit vectors in the sky.

    Parameters
    ----------
    nside : int
        Healpix nside. Should be a power of 2.

    Returns
    -------
    array (npix, 3)
        Unit vectors for each healpix direction.
    """
    npix = hp.nside2npix(nside)
    ipix = jnp.arange(npix)
    x, y, z = hp.pix2vec(nside, ipix)
    res = jnp.asarray([x, y, z]).T
    chex.assert_shape(res, (npix, 3))
    return res
