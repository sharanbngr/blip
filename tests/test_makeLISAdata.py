from jax import numpy as jnp
import chex

from blip.src.makeLISAdata import cholesky_symm


def test_cholesky_symm():
    a = jnp.array([2.0, 1.0])
    b = jnp.array([0.1 + 0.46j, -0.5])
    bbar = b.conj()

    m = jnp.array(
        [
            [a, b, b],
            [bbar, a, b],
            [bbar, bbar, a],
        ]
    )

    cho = cholesky_symm(m)

    # move tdi channel dimensions last to allow matrix product with '@'
    _cho = cho.transpose(2, 0, 1)
    _m = m.transpose(2, 0, 1)
    chex.assert_shape([_cho, _m], (2, 3, 3))

    # hermitean conjugate
    _choH = _cho.conj().transpose(0, 2, 1)

    assert jnp.allclose(_cho @ _choH, _m)
