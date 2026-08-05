"""
The hammer interface, without HAMMER.

Only `HammerCacher` actually talks to HAMMER; everything downstream of it --
the multi-file cacher, the background cacher, the wrappers, the templates and
the fitter -- is ordinary Python over a cacher-shaped object. All of that is
covered here against stand-ins, so it runs in the normal test suite rather than
only where HAMMER happens to be built. `test_hammer_cacher.py` covers the rest.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from redist import modifier_hammer

dir_path = os.path.dirname(__file__)


class DummyCacher:
    """A `HammerCacher` stand-in with fixed bin contents."""

    def __init__(
        self, scaleFactor, nobs, strides, wcs, ffs, norm_factor, element_values
    ):
        self._scaleFactor = scaleFactor
        self._nobs = nobs
        self._strides = strides
        self._wcs = wcs
        self._FFs = ffs
        self._normFactor = norm_factor
        self.element_values = element_values

    def getHistoElementByPosNoScale(self, pos, wcs, FFs):
        return self.element_values[pos]

    def getHistoArray(self, wcs, FFs):
        return np.asarray(self.element_values, dtype=float)

    def getHistoTotal(self):
        return self._normFactor

    def getHistoTotalSM(self):
        return self._normFactor


### TEST MULTIHAMMERCACHER


class TestMultiHammerCacher:
    @pytest.fixture
    def cachers(self):
        return [
            DummyCacher(
                1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 100, [10, 20, 30]
            ),
            DummyCacher(
                1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 200, [15, 25, 35]
            ),
        ]

    @pytest.fixture
    def multi_cacher(self, cachers):
        return modifier_hammer.MultiHammerCacher(cachers)

    def test_multihammer_constructor(self, multi_cacher):
        assert multi_cacher._scaleFactor == 1.0
        assert multi_cacher._nobs == 3
        assert multi_cacher._strides == [1, 2, 3]
        assert multi_cacher._wcs == {"WC1": 1.0}
        assert multi_cacher._FFs == {"FF1": 1.0}
        assert multi_cacher._normFactor == 300
        assert len(multi_cacher._cacherList) == 2

    def test_multihammer_getHistoElementByPos(self, multi_cacher):
        result = multi_cacher.getHistoElementByPos(1, {"WC1": 1.0}, {"FF1": 1.0})

        # scale factor (1.0) over norm factor (300)
        assert result == (20 + 25) * 1.0 / 300

    def test_multihammer_getHistoElementByPosSM(self, multi_cacher):
        result = multi_cacher.getHistoElementByPosSM(1, {"WC1": 1.0}, {"FF1": 1.0})

        assert result == (20 + 25) * 1.0 / 300

    def test_getHistoElementByPosSM_leaves_the_caller_dict_alone(self, multi_cacher):
        """
        Switching the coefficients off must not edit the caller's dictionary.

        It belongs to a wrapper and is regularly shared between templates, so
        zeroing it in place -- as this once did -- switched off the coefficients
        of every other template holding the same dictionary.
        """
        wcs = {"SM": 1.0, "WC1": 5.0}

        multi_cacher.getHistoElementByPosSM(1, wcs, {"FF1": 1.0})

        assert wcs == {"SM": 1.0, "WC1": 5.0}

    def test_getHistoArray_matches_the_per_bin_reads(self, multi_cacher):
        wcs, FFs = {"WC1": 1.0}, {"FF1": 1.0}

        bulk = multi_cacher.getHistoArray(wcs, FFs)

        per_bin = [multi_cacher.getHistoElementByPos(i, wcs, FFs) for i in range(3)]
        assert bulk == pytest.approx(per_bin)

    def test_getHistoArraySM_matches_the_per_bin_reads(self, multi_cacher):
        wcs, FFs = {"SM": 1.0, "WC1": 5.0}, {"FF1": 1.0}

        bulk = multi_cacher.getHistoArraySM(dict(wcs), FFs)

        per_bin = [
            multi_cacher.getHistoElementByPosSM(i, dict(wcs), FFs) for i in range(3)
        ]
        assert bulk == pytest.approx(per_bin)


### TEST BACKGROUNDCACHER


class TestBackgroundCacher:
    @pytest.fixture(scope="class")
    def bkg_file_path(self):
        return os.path.join(dir_path, "hammer_file", "test_hammer_file.dat")

    @pytest.fixture
    def cacher(self, bkg_file_path):
        return modifier_hammer.BackgroundCacher(bkg_file_path, "histo", [1, 2, 3])

    def test_valid_histogram(self, cacher, bkg_file_path):
        assert cacher._fileName == bkg_file_path
        assert cacher._histoName == "histo"
        assert cacher._strides == [1, 2, 3]
        assert np.array_equal(
            cacher._histo, np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        )
        assert cacher._nobs == 10
        assert cacher._normFactor == 550  # sum of [10, 20, ..., 100]

    def test_getHistoElementByPos(self, cacher):
        assert cacher.getHistoElementByPos(2, {}, {}) == 30 / 550

    def test_getHistoArray_matches_the_per_bin_reads(self, cacher):
        bulk = cacher.getHistoArray({}, {})

        per_bin = [cacher.getHistoElementByPos(i, {}, {}) for i in range(10)]
        assert bulk == pytest.approx(per_bin)

    def test_a_missing_file_raises(self, tmp_path):
        """
        Rather than printing and returning half-built.

        Returning from __init__ left an object without `_nobs` or
        `_normFactor`, so the real fault surfaced much later, somewhere
        unrelated, as an AttributeError.
        """
        with pytest.raises(OSError):
            modifier_hammer.BackgroundCacher(str(tmp_path / "absent.dat"), "h", [1])

    def test_an_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.dat"
        empty.write_text("")

        with pytest.raises(ValueError, match="empty"):
            modifier_hammer.BackgroundCacher(str(empty), "h", [1])


### TEST HAMMERNUISWRAPPER, HAMMERNUISWRAPPERSM AND BACKGROUNDNUISWRAPPER


class MockHammerCacher:
    """A cacher whose every bin holds 100, whatever it is asked."""

    def __init__(self):
        self._nobs = 10
        self._wcs = {
            "SM": 1.0,
            "S_qLlL": 1.0,
            "S_qRlL": 1.0,
            "V_qLlL": 1.0,
            "V_qRlL": 1.0,
            "T_qLlL": 1.0,
        }
        self._FFs = {"FF1": 1.0, "FF2": 1.0}
        self._strides = [1, 2, 3]

    def getHistoElementByPos(self, pos, wcs, FFs):
        return 100.0

    def getHistoElementByPosSM(self, pos, wcs, FFs):
        return 100.0

    def getHistoArray(self, wcs, FFs):
        return np.full(self._nobs, 100.0)

    def getHistoArraySM(self, wcs, FFs):
        return np.full(self._nobs, 100.0)


COMPLEX_WCS = {
    "SM": 1.0,
    "Re_S_qLlL": 1.0,
    "Im_S_qLlL": 2.0,
    "Re_S_qRlL": 1.0,
    "Im_S_qRlL": 2.0,
    "Re_V_qLlL": 1.0,
    "Im_V_qLlL": 2.0,
    "Re_V_qRlL": 1.0,
    "Im_V_qRlL": 2.0,
    "Re_T_qLlL": 1.0,
    "Im_T_qLlL": 2.0,
}


@pytest.mark.parametrize(
    "wrapper_type", ["HammerNuisWrapper", "HammerNuisWrapperSM"], indirect=False
)
class TestHammerNuisWrappers:
    """
    Both hammer wrappers, which differ only in whether the coefficients apply.

    Everything they hold and every setter is shared, so it is tested once
    against both rather than written out twice, as it used to be.
    """

    @pytest.fixture
    def mock_hac(self):
        return MockHammerCacher()

    def build(self, wrapper_type, hac, **kwargs):
        return getattr(modifier_hammer, wrapper_type)(hac, **kwargs)

    def test_initialization(self, wrapper_type, mock_hac):
        wrapper = self.build(wrapper_type, mock_hac, param1=2, param2=3)

        assert wrapper._hac == mock_hac
        assert wrapper._nobs == 10
        assert wrapper._wcs == mock_hac._wcs
        assert wrapper._FFs == mock_hac._FFs
        assert wrapper._params == {"param1": 2, "param2": 3}
        assert wrapper._strides == mock_hac._strides
        assert wrapper._dim == len(mock_hac._strides)
        assert wrapper._nbin == 0

    def test_set_wcs(self, wrapper_type, mock_hac):
        wrapper = self.build(wrapper_type, mock_hac, param1=2)

        wrapper.set_wcs(dict(COMPLEX_WCS))

        assert wrapper._wcs == {
            "SM": 1.0,
            "S_qLlL": complex(1.0, 2.0),
            "S_qRlL": complex(1.0, 2.0),
            "V_qLlL": complex(1.0, 2.0),
            "V_qRlL": complex(1.0, 2.0),
            "T_qLlL": complex(1.0, 2.0),
        }

    def test_set_FFs(self, wrapper_type, mock_hac):
        wrapper = self.build(wrapper_type, mock_hac, param1=2, param2=3)

        wrapper.set_FFs({"FF1": 3.0, "FF2": 4.0})

        assert wrapper._FFs == {"FF1": 3.0, "FF2": 4.0}

    def test_set_FFs_drops_unknown_names(self, wrapper_type, mock_hac):
        wrapper = self.build(wrapper_type, mock_hac, param1=2)

        wrapper.set_FFs({"FF1": 3.0, "nonsense": 4.0})

        assert wrapper._FFs == {"FF1": 3.0}

    def test_set_params(self, wrapper_type, mock_hac):
        wrapper = self.build(wrapper_type, mock_hac, param1=2, param2=3)

        wrapper.set_params({"param1": 5, "param2": 6})

        assert wrapper._params == {"param1": 5, "param2": 6}

    def test_set_nbin(self, wrapper_type, mock_hac):
        wrapper = self.build(wrapper_type, mock_hac, param1=2, param2=3)

        wrapper.set_nbin(5)

        assert wrapper._nbin == 5

    def test_evaluate(self, wrapper_type, mock_hac):
        wrapper = self.build(wrapper_type, mock_hac, param1=2, param2=3)
        wrapper.set_nbin(1)
        wrapper.set_wcs(dict(COMPLEX_WCS))
        wrapper.set_FFs({"FF1": 1.0, "FF2": 2.0})

        # the bin holds 100.0, scaled by both nuisance parameters
        assert wrapper.evaluate() == 600.0

    def test_evaluate_all_matches_the_per_bin_loop(self, wrapper_type, mock_hac):
        """The bulk read is the same numbers as walking the bins one by one."""
        wrapper = self.build(wrapper_type, mock_hac, param1=2, param2=3)
        wrapper.set_wcs(dict(COMPLEX_WCS))
        wrapper.set_FFs({"FF1": 1.0, "FF2": 2.0})

        bulk = wrapper.evaluate_all()

        per_bin = []
        for i in range(wrapper._nobs):
            wrapper.set_nbin(i)
            per_bin.append(wrapper.evaluate())
        assert bulk == pytest.approx(per_bin)

    def test_neither_wrapper_is_an_instance_of_the_other(self, wrapper_type, mock_hac):
        """
        They are siblings, not parent and child.

        Anything switching on the wrapper type has to keep seeing what it saw
        when the two classes were written out separately.
        """
        wrapper = self.build(wrapper_type, mock_hac)
        other = (
            modifier_hammer.HammerNuisWrapperSM
            if wrapper_type == "HammerNuisWrapper"
            else modifier_hammer.HammerNuisWrapper
        )

        assert not isinstance(wrapper, other)


class TestBackgroundNuisWrapper:
    @pytest.fixture
    def mock_bkg(self):
        class MockBackgroundCacher:
            def __init__(self):
                self._nobs = 10
                self._wcs = {"SM": 1.0, "S_qLlL": 1.0}
                self._FFs = {"FF1": 1.0, "FF2": 1.0}
                self._strides = [1, 2, 3]

            def getHistoElementByPos(self, pos, wcs, FFs):
                return 100.0

            def getHistoArray(self, wcs, FFs):
                return np.full(self._nobs, 100.0)

        return MockBackgroundCacher()

    def test_initialization(self, mock_bkg):
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        assert wrapper._bkg == mock_bkg
        assert wrapper._nobs == 10
        assert wrapper._wcs == {}
        assert wrapper._FFs == {}
        assert wrapper._params == {"param1": 2, "param2": 3}
        assert wrapper._strides == mock_bkg._strides
        assert wrapper._dim == len(mock_bkg._strides)
        assert wrapper._nbin == 0

    def test_set_nbin(self, mock_bkg):
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        wrapper.set_nbin(5)

        assert wrapper._nbin == 5

    def test_set_wcs_is_ignored(self, mock_bkg):
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        wrapper.set_wcs({"SM": 1.0, "S_qLlL": 1.0})

        assert wrapper._wcs == {}

    def test_set_FFs_is_ignored(self, mock_bkg):
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        wrapper.set_FFs({"FF1": 3.0, "FF2": 4.0})

        assert wrapper._FFs == {}

    def test_set_params(self, mock_bkg):
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        wrapper.set_params({"param1": 5, "param2": 6})

        assert wrapper._params == {"param1": 5, "param2": 6}

    def test_evaluate(self, mock_bkg):
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)
        wrapper.set_nbin(1)
        wrapper.set_wcs({"SM": 1.0, "S_qLlL": 1.0})
        wrapper.set_FFs({"FF1": 1.0, "FF2": 2.0})

        assert wrapper.evaluate() == 600.0

    def test_evaluate_all_matches_the_per_bin_loop(self, mock_bkg):
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        bulk = wrapper.evaluate_all()

        per_bin = []
        for i in range(wrapper._nobs):
            wrapper.set_nbin(i)
            per_bin.append(wrapper.evaluate())
        assert bulk == pytest.approx(per_bin)


### TEST TEMPLATE


class MockWrapper:
    """A wrapper whose bins report the standard model coefficient."""

    def __init__(self):
        self._nobs = 10
        self._wcs = {
            "SM": 1.0,
            "S_qLlL": 1.0,
            "S_qRlL": 1.0,
            "V_qLlL": 1.0,
            "V_qRlL": 1.0,
            "T_qLlL": 1.0,
        }
        self._FFs = {"FF1": 1.0, "FF2": 1.0}
        self._params = {"param1": 1.0, "param2": 2.0}
        self._strides = [1, 2, 3]

    def set_wcs(self, wcs):
        self._wcs = wcs

    def set_FFs(self, FFs):
        self._FFs = FFs

    def set_params(self, params):
        self._params = params

    def set_nbin(self, nbin):
        self._nbin = nbin

    def evaluate(self):
        return 1.0 * self._wcs["SM"]


class TestTemplateClass:
    @pytest.fixture
    def mock_wrapper(self):
        return MockWrapper()

    def test_initialization(self, mock_wrapper):
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        assert obj._name == "TestTemplate"
        assert obj._wrap == mock_wrapper
        assert obj._nobs == 10
        assert obj._nwcs == len(mock_wrapper._wcs)
        assert obj._nFFs == len(mock_wrapper._FFs)
        assert obj._nparams == len(mock_wrapper._params)
        assert obj._strides == mock_wrapper._strides

    def test_generate_template(self, mock_wrapper):
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        bin_contents = obj.generate_template(
            SM=1.0,
            S_qLlL=1.1,
            S_qRlL=1.2,
            V_qLlL=1.3,
            V_qRlL=1.4,
            T_qLlL=1.5,
            FF1=2.0,
            FF2=3.0,
            param1=4.0,
            param2=5.0,
        )

        assert len(bin_contents) == 10
        assert np.all(bin_contents == 1.0)

    def test_generate_template_with_different_params(self, mock_wrapper):
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        bin_contents = obj.generate_template(
            SM=2.0,
            S_qLlL=1.0,
            S_qRlL=1.0,
            V_qLlL=1.0,
            V_qRlL=1.0,
            T_qLlL=1.0,
            FF1=3.0,
            FF2=4.0,
            param1=2.0,
            param2=3.0,
        )

        assert len(bin_contents) == 10
        assert np.all(bin_contents == 2.0)  # evaluate returns 1.0 * SM

    def test_generate_toy(self, mock_wrapper):
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        bin_contents = obj.generate_toy(
            SM=1.0,
            S_qLlL=1.1,
            S_qRlL=1.2,
            V_qLlL=1.3,
            V_qRlL=1.4,
            T_qLlL=1.5,
            FF1=2.0,
            FF2=3.0,
            param1=4.0,
            param2=5.0,
        )

        assert len(bin_contents) == 10
        assert np.all(bin_contents >= 0)  # Poisson draws are non-negative
        assert isinstance(bin_contents[0], np.float64)
        assert np.any(bin_contents != 10.0)

    def test_generate_toy_with_different_params(self, mock_wrapper):
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        bin_contents = obj.generate_toy(
            SM=2.0,
            S_qLlL=1.0,
            S_qRlL=1.0,
            V_qLlL=1.0,
            V_qRlL=1.0,
            T_qLlL=1.0,
            FF1=3.0,
            FF2=4.0,
            param1=2.0,
            param2=3.0,
        )

        assert len(bin_contents) == 10
        assert np.all(bin_contents >= 0)
        assert isinstance(bin_contents[0], np.float64)

    def test_keywords_are_routed_by_name_not_position(self):
        """
        Reordering the call must not change what the values mean.

        The routing used to slice the keywords at offsets derived from the
        number of coefficients, so it depended entirely on the order the caller
        happened to pass them in.
        """
        forwards = modifier_hammer.template("t", MockHammerCacherWrapper())
        backwards = modifier_hammer.template("t", MockHammerCacherWrapper())

        kwargs = dict(
            SM=1.0,
            Re_S_qLlL=0.4,
            Im_S_qLlL=0.2,
            FF1=0.3,
            FF2=0.1,
            lumi=2.0,
        )
        reversed_kwargs = dict(reversed(list(kwargs.items())))

        assert forwards.generate_template(**kwargs) == pytest.approx(
            backwards.generate_template(**reversed_kwargs)
        )

    def test_each_keyword_reaches_its_own_group(self):
        """A coefficient, a form factor and a nuisance must not be confused."""
        wrapper = MockHammerCacherWrapper()
        obj = modifier_hammer.template("t", wrapper)

        obj.generate_template(
            SM=1.0, Re_S_qLlL=0.4, Im_S_qLlL=0.2, FF1=0.3, FF2=0.1, lumi=2.0
        )

        assert wrapper.seen_wcs == {"SM": 1.0, "S_qLlL": complex(0.4, 0.2)}
        assert wrapper.seen_FFs == {"FF1": 0.3, "FF2": 0.1}
        assert wrapper.seen_params == {"lumi": 2.0}

    def test_a_wrapper_without_the_bulk_read_still_works(self, mock_wrapper):
        """
        `MockWrapper` offers only `evaluate`, as any wrapper written before the
        bulk read does, and `template` has to fall back to walking the bins.
        """
        assert not hasattr(mock_wrapper, "evaluate_all")

        obj = modifier_hammer.template("t", mock_wrapper)

        assert len(obj.generate_template(SM=3.0)) == 10


class MockHammerCacherWrapper(modifier_hammer.HammerNuisWrapper):
    """A real wrapper over a mock cacher, recording what each setter was given."""

    def __init__(self):
        super().__init__(MockHammerCacher(), lumi=1.0)
        self._FFs = {"FF1": 0.0, "FF2": 0.0}
        self.seen_wcs = self.seen_FFs = self.seen_params = None

    def set_wcs(self, wcs):
        super().set_wcs(wcs)
        self.seen_wcs = self._wcs

    def set_FFs(self, FFs):
        super().set_FFs(FFs)
        self.seen_FFs = self._FFs

    def set_params(self, params):
        super().set_params(params)
        self.seen_params = self._params


### TEST FITTER


class TestFitterClass:
    @pytest.fixture
    def mock_template(self):
        class MockTemplate:
            def __init__(self):
                self._nobs = 10

            def generate_template(self, **kwargs):
                return np.ones(self._nobs) * 10

        return MockTemplate()

    def test_initialization(self, mock_template):
        obj = modifier_hammer.fitter([mock_template])

        assert obj._template_list == [mock_template]
        assert np.array_equal(obj._data, np.array([]))

    def test_get_template(self, mock_template):
        obj = modifier_hammer.fitter([mock_template])

        assert obj.get_template(0) == mock_template

    def test_upload_data(self, mock_template):
        obj = modifier_hammer.fitter([mock_template])
        data = np.array([1, 2, 3, 4, 5])

        obj.upload_data(data)

        assert np.array_equal(obj._data, data)
        assert obj._data.shape == data.shape

    def test_generate_template_integration(self, mock_template):
        obj = modifier_hammer.fitter([mock_template])

        obj.upload_data(np.array([1, 2, 3, 4, 5]))
        generated_template = obj.get_template(0).generate_template(SM=1.0)

        assert len(generated_template) == 10
        assert np.all(generated_template == 10)


### TEST THE OPTIONAL DEPENDENCY


class TestOptionalDependency:
    """
    HAMMER must stay optional.

    It is not on PyPI and has to be built from source, so importing this module
    may not require it, and only the one class that reads a hammer file may.
    """

    def test_the_module_imports_without_hammer(self):
        # it already did, at the top of this file, on whichever machine is
        # running -- with or without HAMMER built
        assert modifier_hammer.Modifier_Hammer is not None

    def test_hammer_is_not_imported_at_module_scope(self):
        source = Path(modifier_hammer.__file__).read_text()

        assert "\nfrom hammer" not in source
        assert "\nimport hammer" not in source

    def test_the_missing_dependency_points_at_the_readme(self, monkeypatch):
        """
        The error a user without HAMMER meets has to say what to do about it.

        Simulated by hiding the module, so this runs the same way whether or
        not HAMMER happens to be built here.
        """
        monkeypatch.setitem(sys.modules, "hammer.hammerlib", None)
        monkeypatch.setitem(sys.modules, "hammer", None)
        monkeypatch.setattr(
            modifier_hammer.importlib.util, "find_spec", lambda name: None
        )

        with pytest.raises(ImportError, match="not installed") as caught:
            modifier_hammer.HammerCacher("f.dat", "h", {}, "set", {}, {}, 1.0)
        assert "README" in str(caught.value)

    def test_a_built_hammer_that_will_not_load_says_so(self, monkeypatch):
        """
        Installed-but-unloadable is a different problem from absent.

        A HAMMER built without its shared libraries on the loader's path fails
        the same import, and calling that "not installed" sends people off to
        build it a second time. CI hit exactly this, with a missing libboost,
        and the message sent the diagnosis the wrong way.
        """
        monkeypatch.setitem(sys.modules, "hammer.hammerlib", None)
        monkeypatch.setitem(sys.modules, "hammer", None)
        monkeypatch.setattr(
            modifier_hammer.importlib.util, "find_spec", lambda name: object()
        )

        with pytest.raises(ImportError, match="installed but its bindings") as caught:
            modifier_hammer.HammerCacher("f.dat", "h", {}, "set", {}, {}, 1.0)
        assert "LD_LIBRARY_PATH" in str(caught.value)
        assert "not installed" not in str(caught.value)
