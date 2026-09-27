import pytest
from hypothesis import given, example, note, strategies as st

import jax
import chex
from jax import numpy as jnp, vmap, jit

from lisaconstants import ASTRONOMICAL_YEAR as YEAR

from blip.src.faster_geometry import (
    mich_response_unconvolved,
    compute_orbits,
    get_ortho_basis_ecliptic_3d,
    get_vecs_all_sky,
    get_arm_orientations,
    ARMLENGTH,
)
from blip.src.faster_geometry.interp import (
    get_response_interpolator,
    DF_SPARSE,
    DT_SPARSE,
    FMAX_SPARSE,
    TOL_INTERP,
)

jax.config.update("jax_enable_x64", True)

# vectorize response on (t, f)
_mru_vect = vmap(mich_response_unconvolved, (0, None, None, None), out_axes=2)
_mru_vec2 = vmap(_mru_vect, (None, 0, None, None), out_axes=2)
mru_vec2 = jit(_mru_vec2)
mru = jit(mich_response_unconvolved)
mru_vecn = jit(vmap(mich_response_unconvolved, (None, None, 0, None)))

HOUR = 3600
DAY = 24 * HOUR


@st.composite
def interp_patch(draw, fmax=10e-3, dt_s=2 * DAY, df_s=1e-3):
    # Interpolation patch: a rectangle of sides (dt_s, df_s) with a point (t, f)
    # somewhere in the middle. fmax is a limit for the frequencies (towards higher
    # frequencies it gets harder to guarantee precision). The default parameters seem
    # enough for 2e-3 precision on response interpolation.

    # rectangle of sparse grid points
    t0 = draw(st.floats(0, 1.5 * YEAR))
    t1 = t0 + dt_s
    f0 = draw(st.floats(df_s, fmax - df_s))
    f1 = f0 + df_s

    # point to be interpolated
    t = draw(st.floats(t0, t1))
    f = draw(st.floats(f0, f1))

    return (t0, t, t1), (f0, f, f1)


@st.composite
def direction(draw):
    theta = draw(st.floats(0, jnp.pi))
    phi = draw(st.floats(0, 2 * jnp.pi))
    cost, sint = jnp.cos(theta), jnp.sin(theta)
    cosp, sinp = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([sint * cosp, sint * sinp, cost])


@given(
    interp_patch(dt_s=DT_SPARSE, df_s=DF_SPARSE, fmax=FMAX_SPARSE),
    direction(),
    st.just(TOL_INTERP),
)
def test_interpolation_response(patch, n, tol):
    (t0, t, t1), (f0, f, f1) = patch

    note(f"t - t0 = {(t - t0)/3600:.1f} hours")
    note(f"{(t-t0)/(t1-t0) = }")
    note(f"f - f0 = {(f - f0)/1e-3:.2f} mHz")
    note(f"{(f-f0)/(f1-f0) = }")

    times = jnp.array([t0, t, t1])
    freqs = jnp.array([f0, f, f1])
    times_s = jnp.array([t0, t1])
    freqs_s = jnp.array([f0, f1])

    orbits = compute_orbits(times)
    orbits_s = compute_orbits(times_s)

    # get reference "correct" response
    resp = mru_vec2(times, freqs, n, orbits)
    chex.assert_shape(resp, (3, 3, 3, 3))
    assert resp.dtype == jnp.complex128

    # get response from interpolation
    resp_s = mru_vec2(times_s, freqs_s, n, orbits_s)
    _interpolate = get_response_interpolator(times, freqs, times_s, freqs_s)
    resp_interp = _interpolate(resp_s)
    chex.assert_shape(resp_interp, (3, 3, 3, 3))
    assert resp_interp.dtype == jnp.complex128

    #### compare ####

    # no interpolation at all for borders
    assert jnp.allclose(resp[:, :, 0, 0], resp_interp[:, :, 0, 0])
    assert jnp.allclose(resp[:, :, 0, 2], resp_interp[:, :, 0, 2])
    assert jnp.allclose(resp[:, :, 2, 0], resp_interp[:, :, 2, 0])
    assert jnp.allclose(resp[:, :, 2, 2], resp_interp[:, :, 2, 2])

    # in the middle, interpolation should give reasonable results
    maxabs = jnp.max(jnp.abs(resp[:, :, 1, 1]))
    absdiff = jnp.max(jnp.abs(resp - resp_interp)[:, :, 1, 1])
    # reldiff = jnp.max(jnp.abs((resp - resp_interp) / resp)[:, :, 1, 1])
    # note(f"max rel diff = {reldiff:.2e}")
    note(f"max abs diff / max abs resp = {absdiff/maxabs:.2e}")
    note(f"{resp[:,:,1,1] = }")
    note(f"{resp_interp[:,:,1,1] = }")
    assert absdiff < tol * maxabs


@given(interp_patch(), st.floats(1, 10), st.floats(1, 10))
def test_interpolation_scalar_affine(patch, a, b):
    # Here we make sure that the interpolation is just simple linear interpolation.
    # For this we are going to interpolate an affine, complex-valued function:
    # g(f,t) = a*f + (1+1j)*b*t.

    (t0, t, t1), (f0, f, f1) = patch
    note(f"{(t-t0)/(t1-t0) = }")
    note(f"{(f-f0)/(f1-f0) = }")

    times = jnp.array([t0, t, t1])
    freqs = jnp.array([f0, f, f1])
    times_s = jnp.array([t0, t1])
    freqs_s = jnp.array([f0, f1])

    def g(f, t):
        return a * f + (1 + 1j) * b * t

    _gvec = vmap(g, (0, None), 0)
    gvec = vmap(_gvec, (None, 0), 1)

    nf_s, nt_s = 2, 2
    g_sparse = gvec(freqs_s, times_s)
    g_sparse = jnp.stack([g_sparse] * 9).reshape(3, 3, nf_s, nt_s)

    # get correct response in the middle
    g_mid = g(f, t)

    # get interpolation result
    _interpolate = get_response_interpolator(times, freqs, times_s, freqs_s)
    g_interp = _interpolate(g_sparse)
    chex.assert_shape(g_interp, (3, 3, 3, 3))

    #### compare ####
    # no interpolation at all for borders
    assert jnp.allclose(g(f0, t0), g_interp[:, :, 0, 0])
    assert jnp.allclose(g(f0, t1), g_interp[:, :, 0, 2])
    assert jnp.allclose(g(f1, t0), g_interp[:, :, 2, 0])
    assert jnp.allclose(g(f1, t1), g_interp[:, :, 2, 2])

    # in the middle, interpolation should be perfect
    note(f"{g_mid = }")
    note(f"{g_interp[0,0,1,1] = }")
    note(f"difference = {g_mid - g_interp[0,0,1,1]}")
    assert jnp.allclose(g_mid, g_interp[:, :, 1, 1])


@given(st.floats(0, YEAR), st.floats(1e-6, 0.1), direction(), st.integers(1, 2))
def test_response_symmetries(t, f, n, roll):
    orbits = compute_orbits(jnp.array([t]))
    response = mru(t, f, n, orbits)

    # result is real and positive
    for c in range(3):
        assert jnp.allclose(response[c, c].imag, 0.0)
        assert jnp.all(response[c, c].real >= 0)

    # hermitean
    for c1 in range(3):
        for c2 in range(3):
            assert jnp.allclose(response[c1, c2], response[c2, c1].conj())

    # swap spacecraft => swap xyz channels
    orbits_swap = compute_orbits(jnp.array([t]), betaphase=-2 * jnp.pi / 3 * roll)
    response_swap = mru(t, f, n, orbits_swap)
    assert jnp.allclose(response_swap, jnp.roll(response, roll, (0, 1)))


@given(st.floats(-jnp.pi / 2, jnp.pi / 2), st.floats(0, 2 * jnp.pi))
def test_ortho_basis(beta, lam):
    enn, ell, emm = get_ortho_basis_ecliptic_3d(lam, beta)
    assert jnp.isclose(jnp.linalg.norm(enn), 1.0)
    assert jnp.isclose(jnp.linalg.norm(ell), 1.0)
    assert jnp.isclose(jnp.linalg.norm(emm), 1.0)
    assert jnp.isclose(jnp.dot(enn, ell), 0.0)
    assert jnp.isclose(jnp.dot(ell, emm), 0.0)
    assert jnp.isclose(jnp.dot(emm, enn), 0.0)
    assert jnp.allclose(jnp.cross(enn, ell), emm)


@given(st.floats(0, YEAR))
def test_arm_orientations(t):
    times = jnp.array([t])
    orbits = compute_orbits(times)
    for sc in range(1, 4):
        u, v = get_arm_orientations(t, sc, orbits)
        chex.assert_shape([u, v], (3,))
        uv = jnp.dot(u, v)
        # cos(60 deg) = 1/2
        assert jnp.allclose(uv, 0.5)

    u1, v1 = get_arm_orientations(t, 1, orbits)
    u2, v2 = get_arm_orientations(t, 2, orbits)
    u3, v3 = get_arm_orientations(t, 3, orbits)
    assert jnp.allclose(v1, -u3)
    assert jnp.allclose(u1, -v2)
    assert jnp.allclose(u2, -v3)


@given(st.floats(0, YEAR))
def test_armlength(t):
    times = jnp.array([t])
    orbits = compute_orbits(times)

    _, sc_pos, _ = orbits
    assert jnp.isclose(jnp.linalg.norm(sc_pos[0, 0] - sc_pos[1, 0]), ARMLENGTH)
    assert jnp.isclose(jnp.linalg.norm(sc_pos[1, 0] - sc_pos[2, 0]), ARMLENGTH)
    assert jnp.isclose(jnp.linalg.norm(sc_pos[2, 0] - sc_pos[0, 0]), ARMLENGTH)


@given(st.floats(0, YEAR))
def test_low_freq_limit(t):
    orbits = compute_orbits(jnp.array([t]))
    f = 1e-4
    allsky = get_vecs_all_sky(nside=8)
    response = mru_vecn(t, f, allsky, orbits)
    chex.assert_shape(response, (allsky.shape[0], 3, 3))

    # normalization: sky- and polarization-average should be 3/20 in low-freq limit
    for c in range(3):
        assert jnp.allclose(response[:, c, c].mean(), 3 / 20, atol=1e-4)

    # for off-diagonal elements, requirement that T response is zero => (xy) = -(xx)/2
    for c1 in range(3):
        for c2 in range(c1 + 1, 3):
            note(f"{c1=}, {c2=}")
            assert jnp.allclose(response[:, c1, c2].mean(), -3 / 40, atol=1e-4)

    # Response in TT, AT, ET should be very small for low frequencies
    xyz2aet_matrix = jnp.array(
        [
            jnp.array([-1, 0, 1]) / jnp.sqrt(2),
            jnp.array([1, -2, 1]) / jnp.sqrt(6),
            jnp.array([1, 1, 1]) / jnp.sqrt(3),
        ]
    )

    response_aet = jnp.array(xyz2aet_matrix @ response @ xyz2aet_matrix.T)
    eps = 1e-8  # this precision gets better for lower f.
    max_resp = jnp.max(jnp.abs(response_aet))
    assert jnp.max(jnp.abs(response_aet[:, 0, 2])) < eps * max_resp
    assert jnp.max(jnp.abs(response_aet[:, 1, 2])) < eps * max_resp
    assert jnp.max(jnp.abs(response_aet[:, 2, 2])) < eps * max_resp
