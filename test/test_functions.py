import pytest
import numpy as np
from redist import modifier


def test_bintegrate():
    assert list(modifier.bintegrate(lambda x: 1, [np.linspace(0, 5, 6)])) == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert list(modifier.bintegrate(lambda x: x, [np.linspace(0, 5, 6)])) == [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
    ]
    assert pytest.approx(
        modifier.bintegrate(lambda x: np.exp(x), [np.linspace(0, 5, 6)]), 1e-8
    ) == [1.71828183, 4.67077427, 12.69648082, 34.51261311, 93.81500907]


class TestQuadratureCache:
    """The Gauss-Legendre grid is cached, so its key has to be exactly right.

    A stale grid would integrate over the wrong points and still return a
    plausible number, which nothing downstream could catch.
    """

    @staticmethod
    def _linear(x, a=1.0):
        return a * x

    @staticmethod
    def _plane(x, y, a=1.0):
        return a * (x + y)

    def test_different_binnings_do_not_share_a_grid(self):
        first = modifier.bintegrate(
            self._linear, [np.array([0.0, 1.0, 2.0])], quad="gauss"
        )
        second = modifier.bintegrate(
            self._linear, [np.array([0.0, 2.0, 4.0])], quad="gauss"
        )

        assert np.asarray(first) == pytest.approx([0.5, 1.5])
        assert np.asarray(second) == pytest.approx([2.0, 6.0])

    def test_different_orders_do_not_share_a_grid(self):
        """Same binning, different node count: the result must not be reused."""
        bins = [np.array([0.0, 1.0, 2.0])]
        low = modifier.bintegrate(self._linear, bins, quad="gauss", order=2)
        high = modifier.bintegrate(self._linear, bins, quad="gauss", order=32)

        # both are exact for a linear integrand, so they must agree
        assert np.asarray(low) == pytest.approx([0.5, 1.5], abs=1e-14)
        assert np.asarray(high) == pytest.approx([0.5, 1.5], abs=1e-14)

    def test_dimensionality_change_does_not_share_a_grid(self):
        one_d = modifier.bintegrate(self._linear, [np.array([0.0, 2.0])], quad="gauss")
        two_d = modifier.bintegrate(
            self._plane, [np.array([0.0, 2.0]), np.array([0.0, 2.0])], quad="gauss"
        )

        assert np.asarray(one_d).ravel().tolist() == pytest.approx([2.0])
        assert np.shape(two_d) == (1, 1)
        assert np.asarray(two_d).ravel().tolist() == pytest.approx([8.0])

    def test_repeated_calls_agree(self):
        """The second call reads the cache; it must match the first."""
        bins = [np.array([0.0, 1.0, 3.0]), np.array([2.0, 5.0])]
        first = np.asarray(modifier.bintegrate(self._plane, bins, quad="gauss"))
        second = np.asarray(modifier.bintegrate(self._plane, bins, quad="gauss"))

        assert np.array_equal(first, second)
        assert first == pytest.approx(
            np.asarray(modifier.bintegrate(self._plane, bins, quad="nquad")), abs=1e-12
        )

    def test_a_function_cannot_corrupt_the_grid(self):
        """Writing to the arguments would poison every later call."""

        def mutating(x, a=1.0):
            x += 1.0
            return a * x

        with pytest.raises(ValueError, match="read-only"):
            modifier.bintegrate(mutating, [np.array([0.0, 1.0])], quad="gauss")


class TestWeightCache:
    """The weight cache is bounded, and being bounded changes no result.

    It earns its keep inside one likelihood evaluation, where the applier asks
    for the same parameter point several times. A scan or a Markov chain never
    asks twice, so an unbounded cache would only grow.
    """

    pars = {
        "a": {
            "inits": (1.0,),
            "bounds": ((0.0, 10.0),),
            "paramset_type": "unconstrained",
        }
    }
    bins = [np.array([2.0, 3.0, 5.0, 6.0])]
    mapping = np.array([[2.0, 2.0, 1.0], [2.0, 6.0, 2.0]])

    @staticmethod
    def _null(x, a=10.0):
        return a

    @staticmethod
    def _alt(x, a=1.0):
        return a * (1 + x)

    def _build(self, **kwargs):
        return modifier.Modifier(
            self.pars, self._alt, self._null, self.mapping, self.bins, **kwargs
        )

    def test_cache_stops_growing(self):
        cmod = self._build(cache_size=8)
        for i in range(200):
            cmod.weight_func({"a": 1.0 + i * 1e-6})

        assert len(cmod.cache) == 8

    def test_eviction_does_not_change_results(self):
        """A dropped entry is recomputed, so it must come back identical."""
        bounded = self._build(cache_size=2)
        roomy = self._build(cache_size=1000)

        points = [{"a": 1.0 + i * 0.5} for i in range(10)]
        for point in points:
            bounded.weight_func(point)

        for point in points:
            assert np.asarray(bounded.weight_func(point)()) == pytest.approx(
                np.asarray(roomy.weight_func(point)()), rel=0.0, abs=0.0
            )

    def test_repeated_point_hits_the_cache(self):
        cmod = self._build()
        first = cmod.weight_func({"a": 2.0})
        second = cmod.weight_func({"a": 2.0})

        assert first is second
        assert len(cmod.cache) == 1

    def test_least_recently_used_is_dropped_first(self):
        cmod = self._build(cache_size=2)
        cmod.weight_func({"a": 1.0})
        cmod.weight_func({"a": 2.0})
        # touching the older point makes the newer one the eviction candidate
        cmod.weight_func({"a": 1.0})
        cmod.weight_func({"a": 3.0})

        remaining = {key[0][1] for key in cmod.cache}
        assert remaining == {1.0, 3.0}

    def test_zero_disables_the_cache(self):
        cmod = self._build(cache_size=0)
        cmod.weight_func({"a": 2.0})
        cmod.weight_func({"a": 2.0})

        assert cmod.cache == {}


class TestCutoff:
    """Both quadrature rules must exclude the same bins.

    The rules are picked per backend, so a cutoff honoured by one and ignored
    by the other would silently change what a plot shows.
    """

    bins = [np.array([0.0, 1.0, 2.0, 3.0])]
    cutoff = ((1.0, 3.0),)  # excludes the first bin

    @staticmethod
    def _linear(x, a=1.0):
        return a * x

    def test_gauss_marks_the_same_bins_as_nquad(self):
        nquad = np.asarray(
            modifier.bintegrate(self._linear, self.bins, cutoff=self.cutoff)
        )
        gauss = np.asarray(
            modifier.bintegrate(
                self._linear, self.bins, cutoff=self.cutoff, quad="gauss"
            )
        )

        assert np.isnan(gauss).tolist() == np.isnan(nquad).tolist()
        assert gauss[~np.isnan(gauss)] == pytest.approx(
            nquad[~np.isnan(nquad)], abs=1e-12
        )

    def test_no_cutoff_leaves_every_bin(self):
        gauss = np.asarray(modifier.bintegrate(self._linear, self.bins, quad="gauss"))
        assert not np.isnan(gauss).any()

    def test_weights_agree_across_rules(self):
        """The modifier keeps NaN out of its own path, but must still exclude."""

        def null_dist(x, a=10.0):
            return a

        def alt_dist(x, a=1.0, h1=1.0, h2=1.0):
            return a * (1 + x * h1 + x**2 * h2)

        pars = {
            "a": {
                "inits": (1.0,),
                "bounds": ((0.0, 10.0),),
                "paramset_type": "unconstrained",
            }
        }
        binning = [np.array([2.0, 3.0, 5.0, 6.0])]
        mapping = np.array([[2.0, 2.0, 1.0], [2.0, 6.0, 2.0]])

        weights = []
        for quad in ("nquad", "gauss"):
            cmod = modifier.Modifier(
                pars,
                alt_dist,
                null_dist,
                mapping,
                binning,
                cutoff=((3.0, 6.0),),
                quad=quad,
            )
            # the modifier must not carry NaN, whichever rule it uses
            assert not np.isnan(np.asarray(cmod._null_safe)).any()
            weights.append(np.asarray(cmod.get_weights({"a": 2.0})))

        assert weights[1] == pytest.approx(weights[0], rel=1e-10)
        # the excluded bin is pinned to one rather than reweighted
        assert weights[0][0] == pytest.approx(1.0)


class TestDegenerateNull:
    """A bin the null distribution does not populate has no defined weight.

    The ratio is undefined there, so the modifier refuses to build rather than
    returning a yield that looks physical but is not.
    """

    pars = {
        "a": {
            "inits": (1.0,),
            "bounds": ((0.0, 10.0),),
            "paramset_type": "unconstrained",
        }
    }
    mapping = np.array([[2.0, 2.0, 1.0], [2.0, 6.0, 2.0]])

    @staticmethod
    def _odd(x, a=1.0):
        """Integrates to exactly zero over any bin symmetric about zero."""
        return a * x

    @staticmethod
    def _alt(x, a=1.0):
        return a * (1 + x)

    def test_zero_bin_is_rejected(self):
        with pytest.raises(ValueError, match="not be physical"):
            modifier.Modifier(
                self.pars,
                self._alt,
                self._odd,
                self.mapping,
                [np.array([-1.0, 1.0, 2.0, 3.0])],
            )

    def test_error_names_the_offending_bin(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            modifier.Modifier(
                self.pars,
                self._alt,
                self._odd,
                self.mapping,
                [np.array([-1.0, 1.0, 2.0, 3.0])],
            )

    def test_bin_excluded_by_the_cutoff_is_fine(self):
        """Only bins the fit actually uses have to be well defined."""
        cmod = modifier.Modifier(
            self.pars,
            self._alt,
            self._odd,
            self.mapping,
            [np.array([-1.0, 1.0, 2.0, 3.0])],
            cutoff=((1.0, 3.0),),
        )
        weights = np.asarray(cmod.get_weights({"a": 2.0}))

        assert np.isfinite(weights).all()
        assert weights[0] == pytest.approx(1.0)

    def test_both_quadrature_rules_reject(self):
        for quad in ("nquad", "gauss"):
            with pytest.raises(ValueError, match="not be physical"):
                modifier.Modifier(
                    self.pars,
                    self._alt,
                    self._odd,
                    self.mapping,
                    [np.array([-1.0, 1.0, 2.0, 3.0])],
                    quad=quad,
                )

    def test_two_dimensional_bins_are_named_correctly(self):
        """The result is transposed, so the reported edges are easy to get wrong."""

        def null_2d(x, y, a=1.0):
            return a * y

        def alt_2d(x, y, a=1.0):
            return a * (1 + x + y)

        with pytest.raises(ValueError) as excinfo:
            modifier.Modifier(
                self.pars,
                alt_2d,
                null_2d,
                np.ones((2, 2, 2)),
                [np.array([0.0, 1.0, 2.0]), np.array([-1.0, 1.0, 2.0])],
            )

        # the zero bins are those whose y range is symmetric about zero
        assert "[0, 1] x [-1, 1]" in str(excinfo.value)
        assert "[1, 2] x [-1, 1]" in str(excinfo.value)


class TestWeightBound:
    """Every bound is applied, including zero and negative ones.

    The bound used to be tested for truth rather than for presence, so a bound
    of zero was accepted and then silently ignored. Zero and below are ordinary
    bounds once `allow_negative_weights` puts weights on both sides of them.
    """

    pars = {
        "a": {
            "inits": (1.0,),
            "bounds": ((0.0, 10.0),),
            "paramset_type": "unconstrained",
        }
    }
    bins = [np.array([2.0, 3.0, 5.0, 6.0])]
    mapping = np.array([[2.0, 2.0, 1.0], [2.0, 6.0, 2.0]])

    @staticmethod
    def _null(x, a=10.0):
        return a

    @staticmethod
    def _alt(x, a=1.0):
        return a * (1 + x)

    @staticmethod
    def _falling(x, a=1.0):
        """Weights of both signs: [0.025, -0.02, -0.065] over these bins."""
        return a * (1 - 0.3 * x)

    def _build(self, dist=None, **kwargs):
        return modifier.Modifier(
            self.pars,
            dist or self._alt,
            self._null,
            self.mapping,
            self.bins,
            **kwargs,
        )

    def test_no_bound_leaves_the_weights_alone(self):
        cmod = self._build()

        assert cmod.weight_bound is None
        assert np.asarray(cmod.get_weights({"a": 1.0})) == pytest.approx(
            [0.35, 0.5, 0.65]
        )

    def test_positive_bound_clips(self):
        cmod = self._build(weight_bound=0.5)

        assert np.asarray(cmod.get_weights({"a": 1.0})) == pytest.approx(
            [0.35, 0.5, 0.5]
        )

    def test_zero_bound_is_applied_not_ignored(self):
        """The regression: a falsy bound used to be dropped on the floor."""
        cmod = self._build(weight_bound=0.0)

        assert np.asarray(cmod.get_weights({"a": 1.0})) == pytest.approx(
            [0.0, 0.0, 0.0]
        )

    def test_negative_bound_is_applied(self):
        cmod = self._build(
            dist=self._falling, weight_bound=-0.03, allow_negative_weights=True
        )

        # only the weight already below the bound is left alone
        assert np.asarray(cmod.get_weights({"a": 1.0})) == pytest.approx(
            [-0.03, -0.03, -0.065]
        )


def test_svd():
    cov = np.identity(10)
    assert (modifier._svd(cov) == cov).all()

    cov = [
        [1.21000e-04, 3.37920e-04, -2.80830e-03],
        [3.37920e-04, 9.21600e-03, 1.72224e-02],
        [-2.80830e-03, 1.72224e-02, 4.76100e-01],
    ]
    pa = modifier._svd(cov)
    cov_test = pa.dot(pa.T)
    for a, b in zip(cov, cov_test):
        assert pytest.approx(a, 1e-8) == b
