from contextlib import nullcontext as does_not_raise
import pytest
from pytest import raises

import os
from blip.config import parse_config, parse_model_spec

blip_root_dir = os.path.dirname(os.path.dirname(__file__))

TRUEVALS = {
    "noise": {"Np": 9e-42, "Na": 3.6e-49},
    "powerlaw_isgwb": {"alpha": 0.667, "omega0": 5e-7},
}


@pytest.mark.parametrize(
    ["filename"],
    [
        ("params_default.ini",),
        ("params_simple.ini",),
        ("params_test.ini",),
    ],
)
def test_default_params(filename):
    parse_config(os.path.join(blip_root_dir, filename), resume=True)


@pytest.mark.parametrize(
    ["specs", "expectation"],
    [
        ("noise", does_not_raise()),
        ("fixednoise", does_not_raise()),
        ("noise-1", raises(ValueError)),
        ("nnnnn", raises(ValueError)),
        ("population", does_not_raise()),
        ("noise+powerlaw_isgwb", does_not_raise()),
        ("noise+population", does_not_raise()),
        ("noise+population+population", raises(ValueError)),
        ("noise+population-1+population-2", does_not_raise()),
        ("something_something", does_not_raise()),  # should raise error in the future!
    ],
)
def test_model_spec_errors(specs, expectation):
    with expectation:
        parse_model_spec(specs, is_injection=True, truevals_all=TRUEVALS)


def test_file_not_found(tmp_path):
    with raises(FileNotFoundError):
        parse_config(tmp_path / "inexistent.ini", resume=False)
