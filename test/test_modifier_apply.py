"""Tests for how the custom modifier's per-bin weights reach the yields.

The applier used to scatter its results with ``np.place``, which is in-place
NumPy and therefore untraceable. These tests pin the behaviour that the
replacement has to reproduce, in particular the case where one modifier is
attached to several samples and the weights must be reused for each of them.
"""

import json
import os

import numpy as np
import pytest
import pyhf

from redist import modifier

dir_path = os.path.dirname(__file__)


def null_dist(x, a=10.0):
    return a


def alt_dist(x, a=1.0, h1=1.0, h2=1.0):
    return a * (1 + x * h1 + x**2 * h2)


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

CUSTOM_MOD = {"name": "theory", "type": "custom", "data": {"expr": "custom_weight_fn"}}


def _build(samples):
    """Build the simple model with the custom modifier on the given samples."""
    cmod = modifier.Modifier(NEW_PARAMS, alt_dist, null_dist, MAPPING_DIST, [BINNING])
    with open(os.path.join(dir_path, "models", "simple_model.json")) as f:
        spec = json.load(f)
    model = modifier.add_to_model(
        pyhf.Model(spec), ["singlechannel"], samples, cmod.expanded_pyhf, CUSTOM_MOD
    )
    return cmod, model


SIGNAL = np.array([5.0, 10.0])
BACKGROUND = np.array([50.0, 60.0])


def _pars(model, **values):
    """Parameter vector with the named parameters set, by name not position."""
    pars = np.array(model.config.suggested_init())
    for name, value in values.items():
        pars[model.config.par_slice(name)] = value
    return pars


def _bin_factors(cmod, model, pars):
    """The modifier's per-bin multiplicative factor, computed independently.

    ``weight_func`` sees only the modifier's own parameters, the same subset
    pyhf's ParamViewer hands it during ``apply``.
    """
    parvals = {
        name: np.asarray(pars)[model.config.par_slice(name)][0]
        for name in cmod.unco_pars
    }
    return np.asarray(cmod.weight_func(parvals)())


class TestSingleSample:
    """The configuration every existing test and example uses."""

    cmod, model = _build(["signal"])

    def test_yields_factorise(self):
        pars = _pars(self.model, mu=2.0, a=2.0)
        pars[self.model.config.par_slice("h_decorrelated[0]")] = -0.2
        pars[self.model.config.par_slice("h_decorrelated[1]")] = -0.2

        factors = _bin_factors(self.cmod, self.model, pars)
        # signal is scaled by mu and the modifier; background is untouched
        expected = 2.0 * SIGNAL * factors + BACKGROUND

        assert list(self.model.expected_actualdata(pars)) == pytest.approx(
            expected, rel=1e-12
        )


class TestMultipleSamples:
    """One modifier on two samples.

    This is the case that forced ``np.place`` to cycle through its source: the
    mask has twice as many True entries as the weight array has bins, so the
    same weights must be applied to both samples.
    """

    cmod, model = _build(["signal", "background"])

    def test_weights_applied_to_every_sample(self):
        pars = _pars(self.model, mu=2.0, a=2.0)

        factors = _bin_factors(self.cmod, self.model, pars)
        # both samples now carry the modifier, so it factors out entirely
        expected = (2.0 * SIGNAL + BACKGROUND) * factors

        assert list(self.model.expected_actualdata(pars)) == pytest.approx(
            expected, rel=1e-12
        )

    def test_differs_from_single_sample(self):
        """Guards against the weights silently not reaching the second sample."""
        _, single_model = _build(["signal"])

        both = np.asarray(
            self.model.expected_actualdata(_pars(self.model, mu=2.0, a=2.0))
        )
        one = np.asarray(
            single_model.expected_actualdata(_pars(single_model, mu=2.0, a=2.0))
        )

        assert not np.allclose(both, one)


def test_scatter_matches_np_place():
    """The gather-based scatter must reproduce ``np.place`` exactly.

    Covers the shapes the applier actually sees, including the cyclic fill.
    """
    rng = np.random.default_rng(0)
    shapes = [
        (1, 2, 1, 2, 1),
        (1, 3, 1, 4, 2),
        (2, 3, 1, 4, 2),
        (2, 3, 5, 4, 2),
        (1, 2, 1, 3, 2),
    ]
    for n_mod, n_samp, batch, n_bins, n_affected in shapes:
        mask = np.zeros((n_mod, n_samp, batch, n_bins), dtype=bool)
        mask[:, :n_affected, :, :] = True
        source = rng.normal(size=(n_mod, n_bins))

        reference = np.ones_like(mask, dtype=float)
        np.place(reference, mask, source)
        reference = np.where(mask, reference, np.ones(mask.shape))

        flat_mask = mask.ravel()
        indices = np.zeros(flat_mask.size, dtype=int)
        indices[flat_mask] = np.arange(int(flat_mask.sum())) % source.size
        gathered = np.ravel(source)[indices].reshape(mask.shape)
        result = np.where(mask, gathered, np.ones(mask.shape))

        assert np.array_equal(result, reference), f"mismatch for {mask.shape}"
