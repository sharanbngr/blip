import pytest
import pathlib

import jax, chex
from jax import numpy as jnp, vmap, jit, lax
import healpy as hp

from blip.src.faster_geometry import (
    compute_orbits,
    get_arm_orientations,
    get_ortho_basis_ecliptic_3d,
    mich_response_unconvolved,
    get_vecs_all_sky,
    calculate_response_functions,
    FSTAR,
)
from blip.config import parse_config
from blip.src.submodel import submodel, SubmodelKind

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def times():
    t_ref = 5e3 + jnp.arange(421) * 1.5e5
    return t_ref[::30]


@pytest.fixture
def orbits(times):
    return compute_orbits(times)


@pytest.fixture
def response(orbits):
    # Vectorize twice: on times and on sky locations.
    _mru_vect = vmap(
        mich_response_unconvolved, in_axes=(0, None, None, None), out_axes=2
    )  # (3, 3, ntimes)
    _mru_vec2 = vmap(
        _mru_vect, in_axes=(None, None, 0, None), out_axes=3
    )  # (3, 3, ntimes, npix)
    mru_vec2 = jit(_mru_vec2)

    times = orbits[0]
    nside = 8
    response = mru_vec2(times, 1e-3, get_vecs_all_sky(nside), orbits)
    assert response.shape == (3, 3, times.shape[0], hp.nside2npix(nside))
    return response


@pytest.fixture
def config():
    test_dir = pathlib.Path(__file__).parent
    return parse_config(test_dir / "params_test_faster_geometry.ini", resume=False)


def test_arm_orientations(orbits):
    times = orbits[0]
    for sc in range(1, 4):
        u, v = vmap(get_arm_orientations, (0, None, None))(times, sc, orbits)
        chex.assert_shape([u, v], (times.shape[0], 3))
        uv = jnp.vecdot(u, v)
        chex.assert_shape(uv, times.shape)
        # cos(60 deg) = 1/2
        assert jnp.allclose(uv, 0.5)

    for t in times:
        u1, v1 = get_arm_orientations(t, 1, orbits)
        u2, v2 = get_arm_orientations(t, 2, orbits)
        u3, v3 = get_arm_orientations(t, 3, orbits)
        assert jnp.allclose(v1, -u3)
        assert jnp.allclose(u1, -v2)
        assert jnp.allclose(u2, -v3)


def test_ortho_basis():
    key = jax.random.key(538)
    key1, key2 = jax.random.split(key)
    del key
    sinbeta = jax.random.uniform(key1, (10,), minval=-1, maxval=1)
    beta = jnp.arcsin(sinbeta)
    lam = jax.random.uniform(key2, (10,), minval=0, maxval=2 * jnp.pi)

    ns, ls, ms = vmap(get_ortho_basis_ecliptic_3d)(lam, beta)
    for enn, ell, emm in zip(ns, ls, ms):
        assert jnp.isclose(jnp.linalg.norm(enn), 1.0)
        assert jnp.isclose(jnp.linalg.norm(ell), 1.0)
        assert jnp.isclose(jnp.linalg.norm(emm), 1.0)
        assert jnp.isclose(jnp.dot(enn, ell), 0.0)
        assert jnp.isclose(jnp.dot(ell, emm), 0.0)
        assert jnp.isclose(jnp.dot(emm, enn), 0.0)
        assert jnp.allclose(jnp.cross(enn, ell), emm)


def test_response(response):

    for c in range(3):
        # result is real and positive
        assert jnp.allclose(response[c, c].imag, 0.0)
        assert jnp.all(response[c, c].real >= 0)
        # sky maximum is constant in time
        peaks = jnp.max(response[c, c].real, axis=1)

        assert jnp.allclose(peaks, jnp.average(peaks), rtol=5e-2)

    for c1 in range(3):
        for c2 in range(3):
            # hermitean
            assert jnp.allclose(response[c1, c2], response[c2, c1].conj())


def test_low_freq_limit():
    t0 = 0.0
    f0 = 1e-5
    orbits = compute_orbits(jnp.array([t0]))
    allsky = get_vecs_all_sky(nside=8)
    mru_vec = jit(vmap(mich_response_unconvolved, (None, None, 0, None)))
    response = mru_vec(t0, f0, allsky, orbits)
    chex.assert_shape(response, (allsky.shape[0], 3, 3))

    # normalization: sky- and polarization-average should be 3/20 in low-freq limit
    for c in range(3):
        assert jnp.allclose(response[:, c, c].mean(), 3 / 20, atol=1e-4)


def test_calculate_response_functions(config):
    params, inj, misc = config

    times = params["tsegmid"]
    freqs = jnp.fft.rfftfreq(params["Npersplice"], 1.0 / params["fs"])[1:]
    ntimes = times.shape[0]
    nfreqs = freqs.shape[0]
    f0 = freqs / (2 * FSTAR)

    submodels_sgwb = []
    for sm_spec in params["model"]:
        sm = submodel(
            params, inj, sm_spec, freqs, f0, times, injection=False, suffix=""
        )
        if sm_spec.kind == SubmodelKind.SPECTRAL_SPATIAL:
            submodels_sgwb.append(sm)

    calculate_response_functions(freqs, times, submodels_sgwb, params)

    for sm in submodels_sgwb:
        assert hasattr(sm, "response_mat")
        assert hasattr(sm, "inj_response_mat")
        if sm.parameterized_map:
            if sm.basis == "sph":
                ndir = (sm.almax + 1) ** 2
            else:
                # TODO test this case, currently too expensive for some reason
                assert False
                ndir = jnp.sum(sm.mask_idx)
            assert sm.response_mat.shape == (3, 3, nfreqs, ntimes, ndir)
        else:
            # fdata_response_mat will only exist if plot_flag is false
            # assert hasattr(sm, "fdata_response_mat")
            assert sm.response_mat.shape == (3, 3, nfreqs, ntimes)


