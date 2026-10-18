"""
`Modifier_Hammer` against ordinary numpy distributions.

The modifier only ever sees two already-binned templates, so it can be driven
without HAMMER at all. These therefore run in the normal test suite; the
numbers below were fixed before any of the recent changes and pin them.
"""

import json
import os
import warnings

import numpy as np
import pyhf
import pytest

from redist import modifier, modifier_hammer

dir_path = os.path.dirname(__file__)


def null_dist(a=10.0):
    return np.array([a, a])


def alt_dist(a=1.0, **kwargs):
    return np.array(
        [
            a * (1 + kwargs["h[0]"] + kwargs["h[1]"]),
            a * (1 + 3 * kwargs["h[0]"] + 9 * kwargs["h[1]"]),
        ]
    )


NEW_PARAMS = {
    "a": {"inits": (1.0,), "bounds": ((0.0, 10.0),), "paramset_type": "unconstrained"},
    "h": {
        "inits": (1.0, 1.0),
        "bounds": (),
        "cov": [[0.5, 0.1], [0.1, 0.5]],
        "paramset_type": "constrained_by_normal",
    },
}


# Test class for Modifier_Hammer
class TestHammerModifier:
    new_params = NEW_PARAMS
    cmod = modifier_hammer.Modifier_Hammer(new_params, alt_dist, null_dist)

    file = dir_path + "/models/simple_model.json"

    with open(file, "r") as f:
        spec = json.load(f)

    model = pyhf.Model(spec)

    custom_mod = {
        "name": "theory",
        "type": "custom",
        "data": {
            "expr": "custom_weight_fn",
        },
    }

    model = modifier.add_to_model(
        model, ["singlechannel"], ["signal"], cmod.expanded_pyhf, custom_mod
    )
    data = [58.0, 85.0] + model.config.auxdata

    fixed = model.config.suggested_fixed()
    fixed[3] = True

    best_fit = pyhf.infer.mle.fit(data, model, fixed_params=fixed)

    def test_set_up_modifier(self):
        assert "custom" in self.cmod.expanded_pyhf

    def test_add_custom_modifier(self):
        assert "h_decorrelated[0]" in self.model.config.par_map
        assert "h_decorrelated[1]" in self.model.config.par_map

    def test_yields(self):
        init = self.model.config.suggested_init()

        init[0] = 4.0
        init[1] = -1.0
        init[2] = 2.0
        assert pytest.approx(list(self.model.expected_actualdata(init)), 1e-8) == [
            58.19089023,
            159.75693534,
        ]

        init[0] = 10.0
        init[1] = -5.0
        init[2] = 5.0
        assert pytest.approx(list(self.model.expected_actualdata(init)), 1e-8) == [
            92.38612788,
            652.79761315,
        ]

    def test_best_fit(self):
        # The two decorrelated nuisance parameters sit at a few times 1e-2, next
        # to a flat minimum, and where the minimiser stops along it depends on
        # the scipy version: 1.15 and 1.18 differ by 3e-5, which is 4e-4
        # relative on those two and would fail a relative tolerance that says
        # nothing about the fit. The absolute tolerance is what constrains them;
        # the relative one still pins the parameters of order one.
        assert self.best_fit == pytest.approx(
            [
                2.09390946,
                0.02796296,
                -0.03985101,
                1.0,
                1.03135326,
                0.98326032,
            ],
            rel=1e-4,
            abs=1e-4,
        )


PARS = {"a": 2.0, "h_decorrelated[0]": 0.5, "h_decorrelated[1]": -0.5}


class TestWeightBound:
    """
    A weight bound is a number, not a flag.

    `weight_bound` used to be tested for truth, which silently dropped a bound
    of zero. Zero is a bound like any other once `allow_negative_weights` puts
    weights on both sides of it.
    """

    def test_a_bound_of_zero_is_honoured(self):
        cmod = modifier_hammer.Modifier_Hammer(
            NEW_PARAMS,
            alt_dist,
            null_dist,
            weight_bound=0.0,
            allow_negative_weights=True,
        )

        assert np.all(cmod.get_weights(PARS) <= 0.0)

    def test_no_bound_leaves_the_weights_alone(self):
        bounded = modifier_hammer.Modifier_Hammer(
            NEW_PARAMS, alt_dist, null_dist, weight_bound=0.01
        )
        unbounded = modifier_hammer.Modifier_Hammer(NEW_PARAMS, alt_dist, null_dist)

        assert np.all(bounded.get_weights(PARS) <= 0.01)
        assert np.any(unbounded.get_weights(PARS) > 0.01)

    def test_negative_weights_are_lifted_to_one_unless_allowed(self):
        def negative_alt(a=1.0, **kwargs):
            return -np.abs(alt_dist(a, **kwargs))

        refused = modifier_hammer.Modifier_Hammer(NEW_PARAMS, negative_alt, null_dist)
        allowed = modifier_hammer.Modifier_Hammer(
            NEW_PARAMS, negative_alt, null_dist, allow_negative_weights=True
        )

        assert np.all(refused.get_weights(PARS) == 1.0)
        assert np.all(allowed.get_weights(PARS) < 0.0)


class TestEmptyNullBins:
    """
    A bin the null template does not populate carries no ratio.

    Dividing by an empty bin gives infinity, not NaN, and only NaN used to be
    caught, so an infinite weight went straight into the yield.
    """

    @staticmethod
    def half_empty_null(a=10.0):
        return np.array([a, 0.0])

    def test_an_empty_null_bin_gives_a_weight_of_one(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmod = modifier_hammer.Modifier_Hammer(
                NEW_PARAMS, alt_dist, self.half_empty_null
            )

        weights = cmod.get_weights(PARS)

        assert np.all(np.isfinite(weights))
        assert weights[1] == 1.0

    def test_an_empty_null_bin_is_reported(self):
        with pytest.warns(UserWarning, match="1 of 2 bins"):
            modifier_hammer.Modifier_Hammer(NEW_PARAMS, alt_dist, self.half_empty_null)

    def test_a_fully_populated_null_says_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            modifier_hammer.Modifier_Hammer(NEW_PARAMS, alt_dist, null_dist)

    def test_a_non_finite_alternative_bin_gives_a_weight_of_one(self):
        def broken_alt(a=1.0, **kwargs):
            return np.array([np.nan, np.inf])

        cmod = modifier_hammer.Modifier_Hammer(NEW_PARAMS, broken_alt, null_dist)

        assert np.all(cmod.get_weights(PARS) == 1.0)


class TestWeightCache:
    """
    The cache is bounded.

    It exists for the repeated calls one likelihood evaluation makes at
    identical parameters. A scan or a Markov chain never revisits a point, so
    an unbounded cache only grows.
    """

    def test_the_cache_stops_growing(self):
        cmod = modifier_hammer.Modifier_Hammer(
            NEW_PARAMS, alt_dist, null_dist, cache_size=4
        )

        for a in range(20):
            cmod.weight_func(dict(PARS, a=float(a)))

        assert len(cmod.cache) == 4

    def test_the_least_recently_used_entry_goes_first(self):
        cmod = modifier_hammer.Modifier_Hammer(
            NEW_PARAMS, alt_dist, null_dist, cache_size=2
        )
        first, second, third = (dict(PARS, a=float(a)) for a in (1, 2, 3))

        cmod.weight_func(first)
        cmod.weight_func(second)
        cmod.weight_func(first)  # touching it again keeps it
        cmod.weight_func(third)

        keys = [dict(k) for k in cmod.cache]
        assert first in keys
        assert second not in keys

    def test_a_cache_size_of_zero_disables_it(self):
        cmod = modifier_hammer.Modifier_Hammer(
            NEW_PARAMS, alt_dist, null_dist, cache_size=0
        )

        cmod.weight_func(PARS)

        assert cmod.cache == {}

    def test_the_cache_returns_the_same_weights(self):
        cmod = modifier_hammer.Modifier_Hammer(NEW_PARAMS, alt_dist, null_dist)

        first = cmod.weight_func(PARS)()
        second = cmod.weight_func(PARS)()

        assert first == pytest.approx(cmod.get_weights(PARS))
        assert second == pytest.approx(first)


class TestBackend:
    """
    HAMMER is a compiled library called outside the tensor graph.

    Nothing here can be traced, so a tracing backend has to be refused with an
    explanation rather than an AttributeError from somewhere inside.
    """

    def test_a_tracing_backend_is_refused(self):
        pytest.importorskip("jax", reason="jax is not installed")

        cmod = modifier_hammer.Modifier_Hammer(NEW_PARAMS, alt_dist, null_dist)
        try:
            pyhf.set_backend("jax")
            with pytest.raises(ValueError, match="numpy backend"):
                cmod.get_weights(PARS)
        finally:
            pyhf.set_backend("numpy")


class TestSaveLoad:
    """Every setting that changes the weights survives a round trip."""

    @staticmethod
    def spec_with(cmod):
        with open(dir_path + "/models/simple_model.json", "r") as f:
            spec = json.load(f)
        model = modifier.add_to_model(
            pyhf.Model(spec),
            ["singlechannel"],
            ["signal"],
            cmod.expanded_pyhf,
            {"name": "theory", "type": cmod.name, "data": {"expr": "custom_weight_fn"}},
        )
        return model.spec

    def test_the_weight_settings_round_trip(self, tmp_path):
        cmod = modifier_hammer.Modifier_Hammer(
            NEW_PARAMS,
            alt_dist,
            null_dist,
            weight_bound=0.0,
            allow_negative_weights=True,
        )
        file = str(tmp_path / "model.json")

        modifier_hammer.save_hammer(file, self.spec_with(cmod), [cmod])
        _, loaded = modifier_hammer.load_hammer(
            file, alt_dist, null_dist, return_modifier=True
        )

        assert loaded[0].weight_bound == 0.0
        assert loaded[0].allow_negative_weights is True
        assert loaded[0].get_weights(PARS) == pytest.approx(cmod.get_weights(PARS))

    def test_a_file_without_the_newer_keys_still_loads(self, tmp_path):
        """
        `allow_negative_weights` was not always written out.

        A file that predates it omits the key, and the modifier has to fall
        back to its own default rather than fail.
        """
        cmod = modifier_hammer.Modifier_Hammer(NEW_PARAMS, alt_dist, null_dist)
        file = tmp_path / "old_model.json"

        modifier_hammer.save_hammer(str(file), self.spec_with(cmod), [cmod])
        saved = json.loads(file.read_text())
        del saved["allow_negative_weights"]
        file.write_text(json.dumps(saved))

        _, loaded = modifier_hammer.load_hammer(
            str(file), alt_dist, null_dist, return_modifier=True
        )

        assert loaded[0].allow_negative_weights is False
