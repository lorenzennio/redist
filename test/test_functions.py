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
