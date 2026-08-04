"""Tests for running the custom modifier on pyhf's jax backend.

jax is an optional dependency, so the whole module is skipped when it is
absent. Nothing here is required for the NumPy path to work.
"""

import json
import math
import os

import numpy as np
import pytest
import pyhf

from redist import modifier

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

dir_path = os.path.dirname(__file__)


def null_dist(x, a=10.0):
    return a


def alt_dist(x, a=1.0, h1=1.0, h2=1.0):
    return a * (1 + x * h1 + x**2 * h2)


def null_dist_2d(x, y, a=1.0, b=1.0):
    return a * x**2 + b * y**2


BINNING = np.array([2, 3, 5, 6])
MAPPING_DIST = np.array([[2.0, 2.0, 1.0], [2.0, 6.0, 2.0]])

NEW_PARAMS = {
    "a": {
        "inits": (1.0,),
        "bounds": ((0.0, 10.0),),
        "paramset_type": "unconstrained",
    },
    "h": {
        "inits": (1.0, 1.0),
        "bounds": (),
        "cov": [[0.5, 0.1], [0.1, 0.5]],
        "paramset_type": "constrained_by_normal",
    },
}

DATA = [58.0, 85.0]
TEST_PARS = [2.0, -0.2, -0.2, 1.0, 1.0, 1.0]


@pytest.fixture
def numpy_backend():
    pyhf.set_backend("numpy")
    yield
    pyhf.set_backend("numpy")


@pytest.fixture
def jax_backend():
    pyhf.set_backend("jax")
    yield
    pyhf.set_backend("numpy")


def _build(**modifier_kwargs):
    """Build the simple model against whichever backend is active."""
    cmod = modifier.Modifier(
        NEW_PARAMS, alt_dist, null_dist, MAPPING_DIST, [BINNING], **modifier_kwargs
    )
    with open(os.path.join(dir_path, "models", "simple_model.json")) as f:
        spec = json.load(f)
    model = modifier.add_to_model(
        pyhf.Model(spec),
        ["singlechannel"],
        ["signal"],
        cmod.expanded_pyhf,
        {"name": "theory", "type": "custom", "data": {"expr": "custom_weight_fn"}},
    )
    return cmod, model


class TestQuadrature:
    """Gauss-Legendre has to agree with the adaptive rule it stands in for."""

    def test_matches_nquad_1d(self, numpy_backend):
        bins = [np.linspace(0.0, 5.0, 6)]
        for func in (lambda x: 1.0, lambda x: x, np.exp):
            reference = modifier.bintegrate(func, bins, quad="nquad")
            gauss = modifier.bintegrate(func, bins, quad="gauss", order=16)
            assert np.asarray(gauss) == pytest.approx(reference, abs=1e-12)

    def test_matches_nquad_2d(self, numpy_backend):
        """The multi-dimensional node reduction is the easiest thing to get wrong."""
        bins = [np.linspace(0.0, 10.0, 5), np.linspace(4.0, 8.0, 4)]
        reference = modifier.bintegrate(null_dist_2d, bins, (2.0, 0.5), quad="nquad")
        gauss = modifier.bintegrate(null_dist_2d, bins, (2.0, 0.5), quad="gauss")

        assert np.shape(gauss) == np.shape(reference)
        assert np.asarray(gauss) == pytest.approx(reference, abs=1e-10)

    def test_rejects_unknown_rule(self, numpy_backend):
        with pytest.raises(ValueError, match="unknown quadrature rule"):
            modifier.bintegrate(null_dist, [BINNING], quad="simpson")


class TestBackendAgreement:
    """The jax path must reproduce what the NumPy path already produces."""

    def test_yields_match(self, numpy_backend):
        _, np_model = _build()
        np_yields = np.asarray(np_model.expected_actualdata(TEST_PARS))

        pyhf.set_backend("jax")
        try:
            _, jax_model = _build()
            jax_yields = np.asarray(
                jax_model.expected_actualdata(jnp.asarray(TEST_PARS))
            )
        finally:
            pyhf.set_backend("numpy")

        assert jax_yields == pytest.approx(np_yields, rel=1e-10)

    def test_logpdf_matches(self, numpy_backend):
        _, np_model = _build()
        np_data = DATA + np_model.config.auxdata
        np_logpdf = float(np.asarray(np_model.logpdf(TEST_PARS, np_data))[0])

        pyhf.set_backend("jax")
        try:
            _, jax_model = _build()
            jax_data = DATA + jax_model.config.auxdata
            jax_logpdf = float(
                np.asarray(
                    jax_model.logpdf(jnp.asarray(TEST_PARS), jnp.asarray(jax_data))
                )[0]
            )
        finally:
            pyhf.set_backend("numpy")

        assert jax_logpdf == pytest.approx(np_logpdf, rel=1e-10)


class TestGradients:
    """The point of the exercise."""

    @staticmethod
    def _logpdf(model):
        data = jnp.asarray(DATA + model.config.auxdata)

        def logpdf(pars):
            return model.logpdf(pars, data)[0]

        return logpdf

    def test_gradient_matches_finite_differences(self, jax_backend):
        _, model = _build()
        logpdf = self._logpdf(model)
        pars = np.array(TEST_PARS)

        grad = np.asarray(jax.grad(logpdf)(jnp.asarray(pars)))

        finite_differences = []
        for i in range(len(pars)):
            step = 1e-6 * max(1.0, abs(pars[i]))
            up, down = pars.copy(), pars.copy()
            up[i] += step
            down[i] -= step
            finite_differences.append(
                float(logpdf(jnp.asarray(up)) - logpdf(jnp.asarray(down))) / (2 * step)
            )

        assert grad == pytest.approx(np.array(finite_differences), rel=1e-5, abs=1e-6)

    def test_gradient_is_finite(self, jax_backend):
        _, model = _build()
        grad = jax.grad(self._logpdf(model))(jnp.asarray(TEST_PARS))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_gradient_finite_with_cutoff_and_bound(self, jax_backend):
        """Masked bins and clipped weights must not leak NaN into the gradient.

        Under jax a NaN in a discarded branch still produces a NaN gradient, so
        excluded bins are kept out of the division rather than masked after it.
        """
        _, model = _build(cutoff=((3.0, 6.0),), weight_bound=1.5)
        grad = jax.grad(self._logpdf(model))(jnp.asarray(TEST_PARS))

        assert bool(jnp.all(jnp.isfinite(grad)))
        # the cutoff and the bound must actually be doing something here
        assert not bool(jnp.all(grad == 0.0))

    def test_jit_matches_eager(self, jax_backend):
        _, model = _build()
        logpdf = self._logpdf(model)
        pars = jnp.asarray(TEST_PARS)

        assert float(jax.jit(logpdf)(pars)) == pytest.approx(
            float(logpdf(pars)), rel=1e-12
        )

    def test_vmap_matches_loop(self, jax_backend):
        _, model = _build()
        logpdf = self._logpdf(model)
        batch = jnp.asarray(
            [
                TEST_PARS,
                [1.5, -0.1, 0.1, 1.0, 1.0, 1.0],
                [3.0, 0.2, -0.3, 1.0, 1.0, 1.0],
            ]
        )

        looped = np.array([float(logpdf(pars)) for pars in batch])
        mapped = np.asarray(jax.vmap(logpdf)(batch))

        assert mapped == pytest.approx(looped, rel=1e-12)

    def test_jitted_gradient_matches(self, jax_backend):
        _, model = _build()
        logpdf = self._logpdf(model)
        pars = jnp.asarray(TEST_PARS)

        eager = np.asarray(jax.grad(logpdf)(pars))
        compiled = np.asarray(jax.jit(jax.grad(logpdf))(pars))

        assert compiled == pytest.approx(eager, rel=1e-12)


class TestQuadSelection:
    def test_auto_picks_gauss_under_jax(self, jax_backend):
        cmod, _ = _build()
        assert cmod.quad == "gauss"

    def test_auto_picks_gauss_under_numpy_too(self, numpy_backend):
        """These distributions take arrays, so nothing forces adaptive quadrature."""
        cmod, _ = _build()
        assert cmod.quad == "gauss"

    def test_auto_falls_back_for_a_pointwise_distribution(self, numpy_backend):
        """An EOS-style observable can only be called one point at a time."""

        def pointwise(x, a=1.0):
            return a * math.exp(-x / 3.0)

        cmod = modifier.Modifier(
            NEW_PARAMS, pointwise, pointwise, MAPPING_DIST, [BINNING]
        )
        assert cmod.quad == "nquad"

    def test_nquad_under_jax_reports_clearly(self, numpy_backend):
        """Building under NumPy then switching is a mistake worth naming."""
        cmod, _ = _build(quad="nquad")
        pyhf.set_backend("jax")
        try:
            with pytest.raises(ValueError, match="cannot trace through adaptive"):
                cmod.get_weights({k: 0.0 for k in cmod.unco_pars})
        finally:
            pyhf.set_backend("numpy")
