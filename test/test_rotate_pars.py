import numpy as np
import pytest
from redist import modifier


def null_dist(x, a=1.0):
    return a


def alt_dist(x, a=1.0):
    return a


def _modifier(new_pars):
    """Minimal Modifier; only the correlated-parameter bookkeeping is exercised."""
    return modifier.Modifier(
        new_pars,
        alt_dist,
        null_dist,
        np.ones((2, 3)),
        [np.array([0.0, 1.0, 2.0, 3.0])],
    )


class TestRotateParsTwoGroups:
    """Two correlated groups with deliberately different covariances.

    `rotate_pars` must use each group's own rotation matrix and central value.
    A leaked loop variable makes every group but the last-iterated one pick up
    the wrong `uvec` and `mean`.
    """

    new_pars = {
        "g": {
            "inits": (1.0, 2.0),
            "bounds": (),
            "cov": [[0.25, 0.0], [0.0, 0.25]],
            "paramset_type": "constrained_by_normal",
        },
        "h": {
            "inits": (10.0, 20.0),
            "bounds": (),
            "cov": [[4.0, 0.0], [0.0, 9.0]],
            "paramset_type": "constrained_by_normal",
        },
    }

    shifts = {
        "g_decorrelated[0]": 0.5,
        "g_decorrelated[1]": -0.5,
        "h_decorrelated[0]": 1.0,
        "h_decorrelated[1]": 2.0,
    }

    cmod = _modifier(new_pars)

    def _expected(self, key):
        """Rotate one group independently, using the SVD covered by test_svd."""
        info = self.new_pars[key]
        uvec = modifier._svd(info["cov"])
        shift = np.array(
            [self.shifts[f"{key}_decorrelated[{n}]"] for n in range(len(info["inits"]))]
        )
        return np.array(info["inits"]) + uvec @ shift

    def test_both_groups_rotated_independently(self):
        rot = self.cmod.rotate_pars(self.shifts)

        for key in ("g", "h"):
            expected = self._expected(key)
            got = [rot[f"{key}[{n}]"] for n in range(len(expected))]
            assert got == pytest.approx(expected, rel=1e-12)

    def test_first_group_not_contaminated_by_second(self):
        """Targets the leak directly: `g` must not be rotated with `h`'s info.

        `h` has a much larger covariance and a far-away mean, so using it for
        `g` shifts the result by O(10) rather than O(0.1).
        """
        rot = self.cmod.rotate_pars(self.shifts)
        g = np.array([rot["g[0]"], rot["g[1]"]])

        wrong = np.array(self.new_pars["h"]["inits"]) + modifier._svd(
            self.new_pars["h"]["cov"]
        ) @ np.array([0.5, -0.5])

        assert not np.allclose(g, wrong)
        assert g == pytest.approx(self._expected("g"), rel=1e-12)

    def test_single_group_unchanged(self):
        """Guards the path all current tests and examples actually use."""
        cmod = _modifier({"g": self.new_pars["g"]})
        shifts = {k: v for k, v in self.shifts.items() if k.startswith("g")}
        rot = cmod.rotate_pars(shifts)

        assert [rot["g[0]"], rot["g[1]"]] == pytest.approx(
            self._expected("g"), rel=1e-12
        )
