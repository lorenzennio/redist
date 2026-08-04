"""
`HammerCacher` against a real HAMMER histogram file.

These are the only tests that need HAMMER itself, which is not on PyPI and has
to be built from source, so they skip when it is absent. Everything else about
the hammer interface is exercised in `test_hammer_interface.py`, which needs no
HAMMER at all. The `hammer.yml` workflow builds it and fails if these skip.
"""

import os

import numpy as np
import pytest

pytest.importorskip("hammer", reason="HAMMER is not installed; see the README")

from redist import modifier_hammer  # noqa: E402

dir_path = os.path.dirname(__file__)

FILE_NAME = dir_path + "/hammer_file/hammer.dat"
HISTO_NAME = "mmiss2_q2_el"
WILSON_SET = "BtoCTauNu"
SCALE_FACTOR = 1.0

FF_SCHEME = {"name": "SchemeBLPRXP", "Process": "BtoD*", "SchemeVar": "BLPRXPVar"}

FORM_FACTORS = {
    "delta_RhoSq": 0.0,
    "delta_cSt": 0.0,
    "delta_chi21": 0.0,
    "delta_chi2p": 0.0,
    "delta_chi3p": 0.0,
    "delta_eta1": 0.0,
    "delta_etap": 0.0,
    "delta_phi1p": 0.0,
    "delta_beta21": 0.0,
    "delta_beta3p": 0.0,
}

WILSON_COEFFICIENTS = {
    "SM": 1.0,
    "S_qLlL": 0.0,
    "S_qRlL": 0.0,
    "V_qLlL": 0.0,
    "V_qRlL": 0.0,
    "T_qLlL": 0.0,
}


def make_cacher(ff_scheme=None):
    """
    A cacher over the test histogram, with its own copy of the coefficients.

    Its own copy because the cacher keeps them as mutable state; sharing the
    module-level dictionaries would let one test disturb the next.
    """
    return modifier_hammer.HammerCacher(
        FILE_NAME,
        HISTO_NAME,
        ff_scheme or dict(FF_SCHEME),
        WILSON_SET,
        dict(FORM_FACTORS),
        dict(WILSON_COEFFICIENTS),
        SCALE_FACTOR,
    )


@pytest.fixture
def cacher():
    return make_cacher()


class TestHammerCacher:
    def test_constructor_initialization(self, cacher):
        assert cacher._histoName == HISTO_NAME
        assert cacher._FFScheme == FF_SCHEME
        assert cacher._WilsonSet == WILSON_SET
        assert cacher._scaleFactor == SCALE_FACTOR
        assert cacher._wcs == WILSON_COEFFICIENTS
        assert cacher._FFs == FORM_FACTORS

    def test_checkWCCache(self, cacher):
        # unchanged coefficients are reported as cached
        assert cacher.checkWCCache(dict(WILSON_COEFFICIENTS)) is True

        # a changed one is not, and is taken up
        changed = dict(WILSON_COEFFICIENTS, SM=0.9)
        assert cacher.checkWCCache(changed) is False
        assert cacher._wcs == changed

    def test_checkFFCache(self, cacher):
        assert cacher.checkFFCache(dict(FORM_FACTORS)) is True

        changed = dict(FORM_FACTORS, delta_RhoSq=1.0)
        assert cacher.checkFFCache(changed) is False
        assert cacher._FFs == changed

    def test_getHistoTotalSM(self, cacher):
        assert pytest.approx(cacher.getHistoTotalSM(), 1e-0) == 93.0

    def test_getHistoTotalSM_leaves_the_coefficients_in_force(self, cacher):
        """
        The SM total must not leave standard model coefficients behind.

        It switches HAMMER to the standard model to take the sum. If it stopped
        there, `_wcs` would still claim the caller's coefficients while HAMMER
        held the standard model ones, and the next cache check would skip a
        reweighting it owed and hand back standard model yields.
        """
        cacher.getHistoElementByPosNoScale(
            35, dict(WILSON_COEFFICIENTS, SM=2.0), dict(FORM_FACTORS)
        )
        before = cacher._histo_array()

        cacher.getHistoTotalSM()

        assert cacher._wcs["SM"] == 2.0
        assert cacher._histo_array() == pytest.approx(before)

    def test_getHistoElementByPosNoScale(self):
        cacher = make_cacher(
            {"name": "SchemeBLPRXP", "Process": "BtoD*", "SchemeVar": "BLPRXP"}
        )

        result = cacher.getHistoElementByPosNoScale(
            35, dict(WILSON_COEFFICIENTS), dict(FORM_FACTORS)
        )

        assert pytest.approx(result, 1e-2) == 3.76

    def test_getHistoArray_matches_the_per_bin_reads(self, cacher):
        """The bulk read is the same numbers, in the same order."""
        wcs, FFs = dict(WILSON_COEFFICIENTS), dict(FORM_FACTORS)

        bulk = cacher.getHistoArray(wcs, FFs)

        assert len(bulk) == cacher._nobs
        per_bin = [
            cacher.getHistoElementByPosNoScale(i, wcs, FFs) for i in range(cacher._nobs)
        ]
        assert bulk == pytest.approx(per_bin)

    def test_getHistoArray_reweights_when_the_coefficients_change(self, cacher):
        """A changed coefficient has to reach the bulk read, not just the cache."""
        sm = cacher.getHistoArray(dict(WILSON_COEFFICIENTS), dict(FORM_FACTORS))
        shifted = cacher.getHistoArray(
            dict(WILSON_COEFFICIENTS, T_qLlL=0.5), dict(FORM_FACTORS)
        )

        assert not shifted == pytest.approx(sm)

    def test_the_SM_reads_leave_the_caller_dict_alone(self, cacher):
        """
        Switching the coefficients off must not edit the caller's dictionary.

        It belongs to a wrapper, and is often shared between templates, so
        zeroing it in place switched off the coefficients of every other
        template that held the same dictionary.
        """
        wcs = dict(WILSON_COEFFICIENTS, T_qLlL=0.5)

        cacher.getHistoElementByPosNoScaleSM(0, wcs, dict(FORM_FACTORS))

        assert wcs["T_qLlL"] == 0.5

    def test_strides_and_nobs_describe_the_histogram(self, cacher):
        shape = cacher._ham.get_histogram_shape(HISTO_NAME)

        assert cacher._nobs == int(np.prod(shape))
        assert len(cacher._strides) == len(shape)
        assert cacher._strides[-1] == 1
