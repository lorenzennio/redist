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
