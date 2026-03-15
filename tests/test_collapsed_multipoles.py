import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-blip-tests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from blip.src.models import Model, submodel
from blip.src.utils import effective_detector_spectrum


def make_grid():
    fs = np.array([3.0e-4, 6.0e-4, 9.0e-4])
    f0 = np.array([1.0e-3, 2.0e-3, 3.0e-3])
    tsegmid = np.array([5.0e3, 1.5e4, 2.5e4])
    return fs, f0, tsegmid


def make_params(model_name, nside=1):
    return {
        'alias': {},
        'tdi_lev': 'xyz',
        'lisa_config': 'orbiting',
        'nside': nside,
        'lmax': 1,
        'fref': 25.0,
        'sph_flag': True,
        'model': model_name,
        'seglen': 1e5,
    }


def make_inj():
    return {
        'truevals': {},
        'sph_flag': False,
        'doInj': 0,
    }


def test_collapsed_template_matches_m_summed_harmonic_response():
    submodel._anisotropic_response_cache.clear()
    fs, f0, tsegmid = make_grid()
    params = make_params('powerlaw_sph_l2')
    inj = make_inj()

    # Build the higher-ell response first so the lower-ell model reuses the
    # cached Gamma_lm basis instead of recomputing a smaller tensor.
    submodel(params, inj, 'powerlaw_sph_l2', fs, f0, tsegmid)
    ell_one_model = submodel(params, inj, 'powerlaw_sph_l1', fs, f0, tsegmid)

    assert ell_one_model.harmonic_response_almax == 2

    ell_indices = [
        ell_one_model.almtoidx(ell_one_model.harmonic_response_almax, 1, m)
        for m in range(-1, 2)
    ]
    expected_template = np.mean(
        np.abs(ell_one_model.harmonic_response_mat[..., ell_indices])**2,
        axis=-1,
    )

    np.testing.assert_allclose(ell_one_model.response_mat, expected_template)
    assert ell_one_model.spatial_parameters == []


def test_single_multipole_models_are_distinct_and_standalone():
    submodel._anisotropic_response_cache.clear()
    fs, f0, tsegmid = make_grid()
    inj = make_inj()

    ell_one_model = submodel(make_params('powerlaw_sph_l1', nside=2), inj, 'powerlaw_sph_l1', fs, f0, tsegmid)
    ell_two_model = submodel(make_params('powerlaw_sph_l2', nside=2), inj, 'powerlaw_sph_l2', fs, f0, tsegmid)

    assert ell_one_model.response_mat.shape == ell_two_model.response_mat.shape == (3, 3, fs.size, tsegmid.size)
    assert np.linalg.norm(ell_one_model.response_mat - ell_two_model.response_mat) > 0

    theta = [0.0, -10.0]
    cov_one = ell_one_model.cov(theta)
    cov_two = ell_two_model.cov(theta)

    assert cov_one.shape == cov_two.shape == (3, 3, fs.size, tsegmid.size)
    assert ell_one_model.spectral_parameters[1] == r'$\log_{10} (B_{1})$'
    assert ell_two_model.spectral_parameters[1] == r'$\log_{10} (B_{2})$'


def test_effective_detector_spectrum_counts_cross_channel_power():
    covariance = np.zeros((3, 3, 2), dtype=complex)
    covariance[0, 1, :] = [3.0, 4.0]
    covariance[1, 0, :] = [3.0, 4.0]

    spectrum = effective_detector_spectrum(covariance)

    np.testing.assert_allclose(spectrum, np.sqrt(2.0) * np.array([3.0, 4.0]))


def test_odd_multipole_summary_is_not_auto_channel_suppressed():
    submodel._anisotropic_response_cache.clear()
    fs, f0, tsegmid = make_grid()
    inj = make_inj()

    ell_one_model = submodel(make_params('powerlaw_sph_l1', nside=2), inj, 'powerlaw_sph_l1', fs, f0, tsegmid)

    mean_covariance = np.mean(ell_one_model.response_mat, axis=-1)
    summary = effective_detector_spectrum(mean_covariance)
    auto_channel = np.abs(mean_covariance[0, 0, :])
    cross_channel = np.abs(mean_covariance[0, 1, :])

    assert np.all(summary > 0)
    assert np.all(cross_channel > auto_channel)


def test_model_composes_multiple_collapsed_multipoles():
    submodel._anisotropic_response_cache.clear()
    fs, f0, tsegmid = make_grid()
    params = make_params('noise+powerlaw_sph_l1+powerlaw_sph_l2')
    inj = make_inj()
    rmat = np.zeros((fs.size, tsegmid.size, 3, 3), dtype='complex')

    model = Model(params, inj, fs, f0, tsegmid, rmat)

    assert model.submodel_names == ['noise', 'powerlaw_sph_l1', 'powerlaw_sph_l2']
    assert 'powerlaw_sph_l1' in model.submodels
    assert 'powerlaw_sph_l2' in model.submodels

    theta = model.prior(np.full(model.Npar, 0.5))
    assert len(theta) == model.Npar
    assert np.isfinite(model.likelihood(theta))
