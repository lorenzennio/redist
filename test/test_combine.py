"""Tests for multi-channel workspaces and `modifier.combine`.

Both were previously uncovered. They matter because the modifier's weights
reach only a subset of the global bins, which is exactly the indexing the
applier has to get right.
"""

import json

import numpy as np
import pytest
import pyhf

from redist import modifier

try:  # jax is an optional dependency
    import jax.numpy as jnp  # noqa: F401

    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False


def null_dist(x, a=10.0):
    return a


def alt_dist(x, a=1.0, h1=1.0, h2=1.0):
    return a * (1 + x * h1 + x**2 * h2)


BINNING = np.array([2.0, 3.0, 5.0, 6.0])
MAP = np.array([[2.0, 2.0, 1.0], [2.0, 6.0, 2.0]])

PARAMS = {
    "a": {"inits": (1.0,), "bounds": ((0.0, 10.0),), "paramset_type": "unconstrained"},
}

CHANNELS = {
    "chan_a": {"signal": [5.0, 10.0], "background": [50.0, 60.0]},
    "chan_b": {"signal": [7.0, 3.0], "background": [40.0, 30.0]},
}


def _channel_spec(name):
    yields = CHANNELS[name]
    return {
        "name": name,
        "samples": [
            {
                "name": "signal",
                "data": list(yields["signal"]),
                "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
            },
            {
                "name": "background",
                "data": list(yields["background"]),
                "modifiers": [
                    {
                        "name": f"uncorr_{name}",
                        "type": "shapesys",
                        "data": [5.0, 12.0],
                    }
                ],
            },
        ],
    }


def _modifier_spec(name):
    """A custom modifier's `type` is the modifier name it was built with."""
    return {
        "name": f"mod_{name}",
        "type": f"theory_{name}",
        "data": {"expr": f"theory_{name}_weight_fn"},
    }


def _build(channels, modified_channels):
    """Model over `channels`, with the custom modifier on `modified_channels`.

    Several custom modifiers on one model need their modifier sets merged
    before any of them is attached; `add_to_model` rebuilds the model each
    time, and rejects a modifier already in the spec but missing from the set.
    """
    cmods = [
        modifier.Modifier(
            PARAMS, alt_dist, null_dist, MAP, [BINNING], name=f"theory_{name}"
        )
        for name in modified_channels
    ]
    modifier_set = {}
    for cmod in cmods:
        modifier_set = {**modifier_set, **cmod.expanded_pyhf}

    model = pyhf.Model({"channels": [_channel_spec(c) for c in channels]})
    for name in modified_channels:
        model = modifier.add_to_model(
            model, [name], ["signal"], modifier_set, _modifier_spec(name)
        )
    return cmods, model


def _factors(cmod, model, pars):
    """The modifier's per-bin factor, computed independently of pyhf."""
    parvals = {
        name: np.asarray(pars)[model.config.par_slice(name)][0]
        for name in cmod.unco_pars
    }
    return np.asarray(cmod.weight_func(parvals)())


def _pars(model, **values):
    pars = np.array(model.config.suggested_init())
    for name, value in values.items():
        pars[model.config.par_slice(name)] = value
    return pars


class TestMultiChannel:
    """One workspace, two channels, modifier on only one of them."""

    def test_only_the_modified_channel_changes(self):
        cmods, model = _build(["chan_a", "chan_b"], ["chan_a"])
        pars = _pars(model, a=3.0, mu=1.0)

        yields = np.asarray(model.expected_actualdata(pars))
        assert model.config.channels == ["chan_a", "chan_b"]

        factors = _factors(cmods[0], model, pars)
        expected_a = np.array(CHANNELS["chan_a"]["signal"]) * factors + np.array(
            CHANNELS["chan_a"]["background"]
        )
        # the unmodified channel keeps its nominal yields exactly
        expected_b = np.array(CHANNELS["chan_b"]["signal"]) + np.array(
            CHANNELS["chan_b"]["background"]
        )

        assert yields[:2] == pytest.approx(expected_a, rel=1e-12)
        assert yields[2:] == pytest.approx(expected_b, rel=1e-12)

    def test_modifier_on_both_channels(self):
        cmods, model = _build(["chan_a", "chan_b"], ["chan_a", "chan_b"])
        pars = _pars(model, a=3.0, mu=1.0)

        yields = np.asarray(model.expected_actualdata(pars))
        for offset, (cmod, channel) in enumerate(zip(cmods, ["chan_a", "chan_b"])):
            factors = _factors(cmod, model, pars)
            expected = np.array(CHANNELS[channel]["signal"]) * factors + np.array(
                CHANNELS[channel]["background"]
            )
            assert yields[2 * offset : 2 * offset + 2] == pytest.approx(
                expected, rel=1e-12
            )

    def test_channel_order_is_respected(self):
        """Weights must land on the modified channel, not just on some channel."""
        _, model_a = _build(["chan_a", "chan_b"], ["chan_a"])
        _, model_b = _build(["chan_a", "chan_b"], ["chan_b"])

        pars_a = _pars(model_a, a=3.0, mu=1.0)
        pars_b = _pars(model_b, a=3.0, mu=1.0)

        yields_a = np.asarray(model_a.expected_actualdata(pars_a))
        yields_b = np.asarray(model_b.expected_actualdata(pars_b))

        nominal_a = np.array(CHANNELS["chan_a"]["signal"]) + np.array(
            CHANNELS["chan_a"]["background"]
        )
        nominal_b = np.array(CHANNELS["chan_b"]["signal"]) + np.array(
            CHANNELS["chan_b"]["background"]
        )

        # modifying chan_a leaves chan_b nominal, and the other way round
        assert yields_a[2:] == pytest.approx(nominal_b, rel=1e-12)
        assert yields_b[:2] == pytest.approx(nominal_a, rel=1e-12)
        assert not np.allclose(yields_a[:2], nominal_a)
        assert not np.allclose(yields_b[2:], nominal_b)


def _save_single_channel(tmp_path, channel):
    """Save a one-channel model with its own custom modifier."""
    cmods, model = _build([channel], [channel])
    data = [
        s + b
        for s, b in zip(CHANNELS[channel]["signal"], CHANNELS[channel]["background"])
    ]
    path = str(tmp_path / f"{channel}.json")
    modifier.save(path, model.spec, cmods, data)
    return path, cmods[0], model, data


class TestSaveLoad:
    def test_roundtrip_preserves_yields(self, tmp_path):
        path, _, model, data = _save_single_channel(tmp_path, "chan_a")

        loaded, cmods, loaded_data = modifier.load(
            path, alt_dist, null_dist, return_modifier=True, return_data=True
        )

        assert loaded_data == data
        pars = _pars(model, a=3.0, mu=1.0)
        assert list(loaded.expected_actualdata(pars)) == pytest.approx(
            list(model.expected_actualdata(pars)), rel=1e-12
        )

    def test_bins_serialise_as_nested_lists(self, tmp_path):
        """`bins` is a list of arrays, one per kinematic dimension."""
        path, _, _, _ = _save_single_channel(tmp_path, "chan_a")
        with open(path) as f:
            saved = json.load(f)

        assert saved["bins"] == [[list(BINNING)]]

    def test_saves_two_dimensional_binning(self, tmp_path):
        def null_2d(x, y, a=1.0, b=1.0):
            return a * x**2 + b * y**2

        bins = [np.linspace(0.0, 10.0, 5), np.linspace(4.0, 8.0, 4)]
        cmod = modifier.Modifier(
            {
                "a": {
                    "inits": (1.0,),
                    "bounds": ((0.0, 5.0),),
                    "paramset_type": "unconstrained",
                }
            },
            null_2d,
            null_2d,
            np.ones((2, 4, 3)),
            bins,
        )
        path = str(tmp_path / "twod.json")
        modifier.save(path, {"channels": []}, [cmod])

        with open(path) as f:
            saved = json.load(f)
        assert saved["bins"] == [[list(b) for b in bins]]


class TestCombine:
    def test_combined_channels_match_the_inputs(self, tmp_path):
        paths, models = [], []
        for channel in ("chan_a", "chan_b"):
            path, _, model, _ = _save_single_channel(tmp_path, channel)
            paths.append(path)
            models.append(model)

        combined, data = modifier.combine(
            paths, [alt_dist, alt_dist], [null_dist, null_dist], return_data=True
        )

        assert combined.config.channels == ["chan_a", "chan_b"]
        assert len(data) == len(combined.config.auxdata) + 4

        # `a` and `mu` are shared, so each channel must reproduce its own model
        pars = _pars(combined, a=3.0, mu=1.0)
        combined_yields = np.asarray(combined.expected_actualdata(pars))

        for offset, model in enumerate(models):
            standalone = np.asarray(
                model.expected_actualdata(_pars(model, a=3.0, mu=1.0))
            )
            assert combined_yields[2 * offset : 2 * offset + 2] == pytest.approx(
                standalone, rel=1e-12
            )

    def test_both_modifiers_stay_active(self, tmp_path):
        paths = [
            _save_single_channel(tmp_path, channel)[0]
            for channel in ("chan_a", "chan_b")
        ]
        combined = modifier.combine(paths, [alt_dist, alt_dist], [null_dist, null_dist])

        nominal = np.asarray(
            combined.expected_actualdata(_pars(combined, a=1.0, mu=1.0))
        )
        shifted = np.asarray(
            combined.expected_actualdata(_pars(combined, a=4.0, mu=1.0))
        )

        # every channel must respond to the shared theory parameter
        assert not np.allclose(nominal[:2], shifted[:2])
        assert not np.allclose(nominal[2:], shifted[2:])


@pytest.mark.skipif(not HAS_JAX, reason="jax is an optional dependency")
class TestJaxAgreement:
    """Multi-channel and combined models must agree across backends."""

    def _yields(self, backend, builder):
        pyhf.set_backend(backend)
        try:
            model, pars = builder()
            return np.asarray(model.expected_actualdata(pars))
        finally:
            pyhf.set_backend("numpy")

    def test_multichannel_matches(self):
        def builder():
            _, model = _build(["chan_a", "chan_b"], ["chan_a"])
            return model, _pars(model, a=3.0, mu=1.0)

        numpy_yields = self._yields("numpy", builder)
        jax_yields = self._yields("jax", builder)
        assert jax_yields == pytest.approx(numpy_yields, rel=1e-10)

    def test_combined_matches(self, tmp_path):
        paths = [
            _save_single_channel(tmp_path, channel)[0]
            for channel in ("chan_a", "chan_b")
        ]

        def builder():
            model = modifier.combine(
                paths, [alt_dist, alt_dist], [null_dist, null_dist]
            )
            return model, _pars(model, a=3.0, mu=1.0)

        numpy_yields = self._yields("numpy", builder)
        jax_yields = self._yields("jax", builder)
        assert jax_yields == pytest.approx(numpy_yields, rel=1e-10)

    def test_combined_gradient_is_finite(self, tmp_path):
        import jax

        paths = [
            _save_single_channel(tmp_path, channel)[0]
            for channel in ("chan_a", "chan_b")
        ]
        pyhf.set_backend("jax")
        try:
            model = modifier.combine(
                paths, [alt_dist, alt_dist], [null_dist, null_dist]
            )
            pars = _pars(model, a=3.0, mu=1.0)
            data = jnp.asarray([55.0, 70.0, 47.0, 33.0] + list(model.config.auxdata))
            grad = jax.grad(lambda p: model.logpdf(p, data)[0])(jnp.asarray(pars))
            assert bool(jnp.all(jnp.isfinite(grad)))
            assert not bool(jnp.all(grad == 0.0))
        finally:
            pyhf.set_backend("numpy")
