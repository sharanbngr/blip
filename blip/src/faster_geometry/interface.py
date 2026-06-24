"""Interface with the rest of BLIP."""

import functools
import logging
from collections import namedtuple

import jax
from jax import numpy as jnp, vmap, jit
import chex
import numpy as np
import healpy as hp
from scipy.special import sph_harm_y

from tqdm import tqdm

from .core import mich_response_unconvolved
from .orbit import compute_orbits
from .interp import get_response_interpolator, get_sparse_tf_grid
from .util import get_vecs_all_sky
from .const import INTERPOLATION_ALLOWED, LOG_PERFORMANCE, FSTAR
from ..submodel import submodel

__all__ = ["calculate_response_functions"]

logger = logging.getLogger(__name__)

_GridInfo = namedtuple(
    "GridInfo",
    ["nf", "nt", "nf_s", "nt_s", "times", "freqs", "times_sparse", "freqs_sparse"],
)
_SkyInfo = namedtuple("SkyInfo", ["nside", "npix", "active_pixels_idx", "dOmega"])


def calculate_response_functions(freqs, times, submodels, params, plot_flag=False):
    """
    Compute SGWB responses for each submodel and write it to the submodel attributes.

    This procedure avoids redundant calculations by only computing the response for a
    given pixel once. It also uses a sparse time-frequency grid and linearly
    interpolates the results in-between.

    It mirrors the functionality in fast_geometry.calculate_response_functions().

    Parameters
    ----------
    freqs : array (nfreqs,)
        frequencies.
    times : array (ntimes,)
        times.
    submodels : list[submodel]
        The submodels whose response matrices will be computed. They must be in pixel
        basis and have a skymap attribute, not assumed normalized.

        These objects also act as output parameters: after the procedure has run, each
        of them will receive a response matrix as an attribute, which is an array of
        shape (3, 3, nfreqs, ntimes).

        The attributes used for the output depend on the value of plot_flag: the
        response matrix will be written to sm.response_mat and sm.inj_response_mat if it
        is False, or to sm.fdata_response_mat if it is True.
    params: dict
        Parameter dictionary from parse_config().
    plot_flag : bool, optional
        If True, don't write the output response matrix to sm.response_mat or
        sm.inj_response_mat, but to submodel.fdata_response_mat. This is useful for
        plotting simulation and analysis models together (hence the name). Defaults to
        False.

    Warnings
    --------
    This is experimental, tested only for healpix responses.
    """
    chex.assert_rank([freqs, times], 1)

    for sm in submodels:
        assert isinstance(sm, submodel)
        assert hasattr(sm, "has_map")
        assert sm.has_map or sm.fullsky
    del sm  # deleted to prevent accidental use out of scope as in issue #28
    assert params["lisa_config"] == "orbiting"

    ############## Set up vectorization, interpolation, orbits ############

    # Vectorize twice: on times and on frequencies.
    # Sky directions are done sequentially
    _mru_vect = vmap(
        mich_response_unconvolved, (0, None, None, None), out_axes=2
    )  # (3, 3, ntimes)
    _mru_vec2 = vmap(
        _mru_vect, (None, 0, None, None), out_axes=2
    )  # (3, 3, nfreqs, ntimes)
    mru_vec2 = jit(_mru_vec2)

    # Set up sparse grid for interpolation
    if INTERPOLATION_ALLOWED:
        logger.info("Using interpolation")
        times_s, freqs_s = get_sparse_tf_grid(times, freqs)
        interpolator = get_response_interpolator(times, freqs, times_s, freqs_s)
        interpolator = jit(interpolator)
    else:
        times_s, freqs_s = times, freqs
        interpolator = lambda x: x  # noqa: E731
    nt, nf, nt_s, nf_s = len(times), len(freqs), len(times_s), len(freqs_s)
    nside = params["nside"]
    npix = hp.nside2npix(nside)
    dOmega = 4 * np.pi / npix
    gridinfo = _GridInfo(nf, nt, nf_s, nt_s, times, freqs, times_s, freqs_s)

    orbits = compute_orbits(times_s)

    ################ compute responses for pixels with nonzero power #############

    active_pixels_idx, active_pixels_vecs = _find_nonzero_pixels(submodels, nside, npix)
    skyinfo = _SkyInfo(nside, npix, active_pixels_idx, dOmega)

    if LOG_PERFORMANCE:
        _log_performance_analysis(mru_vec2, gridinfo, skyinfo, orbits)

    print("Computing LISA response functions...")
    resp_s = []
    for i, pix_vec in tqdm(
        zip(active_pixels_idx, active_pixels_vecs),
        total=len(active_pixels_idx),
        desc="response",
        unit="pixel",
    ):
        # We block on this computation before dispatching the next one to the GPU,
        # otherwise the progress bar is meaningless. A possible optimization would be
        # to dispatch the next iteration before blocking on the current one.
        _r = mru_vec2(times_s, freqs_s, pix_vec, orbits)
        resp_s.append(_r.block_until_ready())

    chex.assert_shape(resp_s[0], (3, 3, nf_s, nt_s))
    chex.assert_equal_shape(resp_s)

    ################# Integrate on sky for each submodel. ################

    # We do this sequentially with a python for loop and using plain numpy (not jax)
    # arrays. The point of this is to avoid allocating memory for the whole integrand, a
    # large rank-5 tensor.
    for sm in submodels:
        # anisotropic submodels (with maps)
        if sm.has_map:
            # submodels with fixed spatial templates (output has 4 dims)
            if not sm.parameterized_map:
                rmat, postf_dims = _do_fixed_submodel(
                    sm, gridinfo, skyinfo, interpolator, resp_s
                )
                chex.assert_shape(rmat, (3, 3, nf, nt))
                assert postf_dims == 1

            # submodels with parameterized spatial templates (output has 5 dims)
            else:
                rmat, postf_dims = _do_parameterized_submodel(
                    sm, gridinfo, skyinfo, interpolator, resp_s
                )
                chex.assert_shape(rmat, (3, 3, nf, nt, None))
                assert postf_dims == 2

        # isotropic SGWB case (effectively a fixed template)
        elif sm.spatial_model_name == "isgwb":
            rmat, postf_dims = _do_isotropic_submodel(
                gridinfo, skyinfo, interpolator, resp_s
            )
            chex.assert_shape(rmat, (3, 3, nf, nt))
            assert postf_dims == 1

        else:
            assert False

        rmat = _mich_to_tdi(rmat, freqs, params, postf_dims)

        # Assert postconditions for clarity: output shape and dtype.
        output_shape = (3, 3, nf, nt)
        if sm.has_map and sm.parameterized_map:
            if sm.basis == "pixel":
                sky_size = len(sm.mask_idx)
            else:
                sky_size = (sm.almax + 1) ** 2
            output_shape = (3, 3, nf, nt, sky_size)
        assert rmat.shape == output_shape
        assert rmat.dtype == complex

        ##### add reference to the response matrix in the submodel  #####
        if plot_flag:
            sm.fdata_response_mat = rmat
        else:
            sm.response_mat = rmat
            sm.inj_response_mat = rmat  # deprecated


def _find_nonzero_pixels(submodels, nside, npix):
    is_fullsky = any([getattr(sm, "fullsky", None) for sm in submodels])
    if is_fullsky:
        active_pixels_idx = jnp.arange(npix)
    else:
        active_pixels_idx_sm = [
            _intarray_to_set(jnp.nonzero(sm.skymap)[0])
            for sm in submodels
            if sm.has_map
        ]
        active_pixels_idx = jnp.array(
            sorted(
                list(functools.reduce(lambda x, y: x.union(y), active_pixels_idx_sm))
            ),
            dtype=int,
        )

    assert active_pixels_idx.dtype == int

    active_pixels_vecs = get_vecs_all_sky(nside)[active_pixels_idx]
    chex.assert_shape(active_pixels_vecs, (None, 3))
    return active_pixels_idx, active_pixels_vecs


def _do_fixed_submodel(sm, gridinfo, skyinfo, interpolator, resp_s):
    nf, nt = gridinfo.nf, gridinfo.nt
    npix, nside = skyinfo.npix, skyinfo.nside
    dOmega = skyinfo.dOmega
    active_pixels_idx = skyinfo.active_pixels_idx

    rmat = np.zeros((3, 3, nf, nt), dtype=complex)
    postf_dims = 1  # nb of tensor dimensions after freqs

    if sm.basis == "pixel":
        chex.assert_shape(sm.skymap, (npix,))

        ## normalize skymap such that it integrates to 1 over the sky
        skymap_normalized = sm.skymap / (jnp.sum(sm.skymap) * dOmega)
        for i, response_sparse in zip(active_pixels_idx, resp_s):
            response = interpolator(response_sparse)
            rmat += skymap_normalized[i] * response * dOmega

    elif sm.basis == "sph":
        alm_size = (sm.almax + 1) ** 2
        ## angular coordinates of pixel indices
        theta, phi = hp.pix2ang(nside, active_pixels_idx)
        Ylms = np.zeros((npix, alm_size), dtype="complex")
        ## Get the spherical harmonics
        for ii in range(alm_size):
            lval, mval = sm.idxtoalm(sm.almax, ii)
            Ylms[:, ii] = sph_harm_y(mval, lval, theta, phi)
            ## check that the Ylms have the right number of pixels, sph terms
        chex.assert_shape(Ylms, (npix, alm_size))

        for i, response_sparse in zip(active_pixels_idx, resp_s):
            response = interpolator(response_sparse)
            # from Ylms, get skymap intensity for this pixel
            pix_sph_sum = np.sum(Ylms[i, :] * sm.alms_inj)
            rmat += pix_sph_sum * response * dOmega

    else:
        assert False

    return rmat, postf_dims


def _do_parameterized_submodel(sm, gridinfo, skyinfo, interpolator, resp_s):
    nf, nt = gridinfo.nf, gridinfo.nt
    npix, nside = skyinfo.npix, skyinfo.nside
    dOmega = skyinfo.dOmega
    active_pixels_idx = skyinfo.active_pixels_idx

    if sm.basis == "pixel":
        sky_size = len(sm.mask_idx)
        rmat = np.zeros((3, 3, nf, nt, sky_size), dtype=complex)
        postf_dims = 2  # nb of tensor dimensions after freqs
        assert np.allclose(sm.mask_idx,active_pixels_idx)
        for j, (i,response_sparse) in enumerate(zip(active_pixels_idx, resp_s)):
            if i in sm.mask_idx:
                response = interpolator(response_sparse)
                rmat[..., j] = response

    elif sm.basis == "sph":
        alm_size = (sm.almax + 1) ** 2
        ## angular coordinates of pixel indices
        theta, phi = hp.pix2ang(nside, active_pixels_idx)
        Ylms = np.zeros((npix, alm_size), dtype="complex")
        ## Get the spherical harmonics
        for ii in range(alm_size):
            lval, mval = sm.idxtoalm(sm.almax, ii)
            Ylms[:, ii] = sph_harm_y(mval, lval, theta, phi)
            ## check that the Ylms have the right number of pixels, sph terms
        chex.assert_shape(Ylms, (npix, alm_size))

        ## 3 x 3 x frequency x time x Ylms
        rmat = np.zeros((3, 3, nf, nt, alm_size), dtype="complex")
        postf_dims = 2  ## time x Ylms

        ## loop over pixels, interpolating the response as we go
        for i, response_sparse in zip(active_pixels_idx, resp_s):
            response = interpolator(response_sparse)
            rmat += Ylms[None, None, None, None, i, :] * response[..., None] * dOmega
    else:
        assert False

    return rmat, postf_dims


def _do_isotropic_submodel(gridinfo, skyinfo, interpolator, resp_s):
    nf, nt = gridinfo.nf, gridinfo.nt
    dOmega = skyinfo.dOmega
    active_pixels_idx = skyinfo.active_pixels_idx

    rmat = np.zeros((3, 3, nf, nt), dtype=complex)
    postf_dims = 1  ## just time

    ## loop over pixels, interpolating the response as we go
    for i, response_sparse in zip(active_pixels_idx, resp_s):
        response = interpolator(response_sparse)
        rmat += (1 / (4 * jnp.pi)) * response * dOmega

    return rmat, postf_dims


def _mich_to_tdi(rmat, freqs, params, postf_dims):
    if params["tdi_lev"] == "xyz":
        mich_to_x1 = 4 * jnp.sin(freqs / FSTAR) ** 2
        rmat = (
            rmat
            * mich_to_x1[
                np.newaxis, np.newaxis, :, *[np.newaxis for i in range(postf_dims)]
            ]
        )
    elif params["tdi_lev"] == "aet":
        # TODO
        raise NotImplementedError

    return rmat


def _log_performance_analysis(mru_vec2, gridinfo, skyinfo, orbits):
    times_sparse = gridinfo.times_sparse
    freqs_sparse = gridinfo.freqs_sparse
    freqs = gridinfo.freqs
    nt_s = gridinfo.nt_s
    nf_s = gridinfo.nf_s

    ## check runtime, memory use
    _compiled = mru_vec2.lower(
        times_sparse, freqs_sparse, jnp.zeros(3), orbits
    ).compile()
    logger.debug("freq range %f - %f", freqs[0], freqs[-1])
    logger.debug("sparse ntimes = %d, sparse nfreqs = %d", nt_s, nf_s)
    logger.debug("response execution cost analysis: %s", _compiled.cost_analysis())
    logger.debug("response memory cost analysis: %s", _compiled.memory_analysis())
    # From a CPU test:
    #    output size =  144 bytes / time / freq / pixel (one complex 3x3 matrix)
    # bytes accessed = 3049 bytes / time / freq / pixel
    #          flops = 1378  flop / time / freq / pixel

    # NOTE(solano) the usage of 3x3 complex matrices in our context is a waste of RAM.
    # We assume equal and constant LTTs, so our SGWB can be perfectly described by two
    # real series: the AA and EE time-varying PSDs. That would reduce the output from
    # 144 bytes to 16 bytes per time per frequency per pixel (9x compression). And
    # that's still using double precision, which we don't need for the response matrix
    # output. Using single precision we could do 144 -> 8 bytes, an 18x compression.
    # NOTE(awc) However, the matrix inversions in the likelihood require double
    # precision, and the spherical harmonic search requires the complex responses.


def _intarray_to_set(a: jax.Array) -> set:
    assert a.dtype == int
    return set([int(v) for v in a.ravel()])
