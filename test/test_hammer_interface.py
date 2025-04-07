import os
import pytest
import numpy as np
from redist import modifier_hammer

dir_path = os.path.dirname(__file__)

#### Test the HammerCacher
class TestHammerCacher:
    def test_constructor_initialization(self):
        # Arrange: Input values
        file_name = dir_path+"/hammer_file/hammer.dat"
        histo_name = "mmiss2_q2_el"
        ff_scheme = {"name":"SchemeBLPRXP","Process":"BtoD*","SchemeVar":"BLPRXPVar"}
        wilson_set = "BtoCTauNu"
        form_factors = {"delta_RhoSq": 0.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}
        wilson_coefficients = {"SM": 1.0,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}
        scale_factor = 1.0

        # Act: Create an instance
        cacher = modifier_hammer.HammerCacher(
            file_name, histo_name, ff_scheme, wilson_set, form_factors, wilson_coefficients, scale_factor
        )

        # Assert: Verify attributes are set correctly
        assert cacher._histoName == histo_name
        assert cacher._FFScheme == ff_scheme
        assert cacher._WilsonSet == wilson_set
        assert cacher._scaleFactor == scale_factor
        assert cacher._wcs == wilson_coefficients
        assert cacher._FFs == form_factors

    def test_checkWCCache(self):
        # Arrange: Create an instance with pre-filled Wilson coefficients
        file_name = dir_path+"/hammer_file/hammer.dat"
        histo_name = "mmiss2_q2_el"
        ff_scheme = {"name":"SchemeBLPRXP","Process":"BtoD*","SchemeVar":"BLPRXPVar"}
        wilson_set = "BtoCTauNu"
        form_factors = {"delta_RhoSq": 0.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}
        wilson_coefficients = {"SM": 1.0,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}
        scale_factor = 1.0

        # Act: Create an instance
        cacher = modifier_hammer.HammerCacher(
            file_name, histo_name, ff_scheme, wilson_set, form_factors, wilson_coefficients, scale_factor
        )

        # Act & Assert: Case 1 - Cache matches input
        assert cacher.checkWCCache({"SM": 1.0,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}) is True

        # Act & Assert: Case 2 - Cache differs (new key added)
        assert cacher.checkWCCache({"SM": 0.9,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}) is False
        assert cacher._wcs == {"SM": 0.9,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}

    def test_checkFFCache(self):
        # Arrange: Create an instance with pre-filled Wilson coefficients
        file_name = dir_path+"/hammer_file/hammer.dat"
        histo_name = "mmiss2_q2_el"
        ff_scheme = {"name":"SchemeBLPRXP","Process":"BtoD*","SchemeVar":"BLPRXPVar"}
        wilson_set = "BtoCTauNu"
        form_factors = {"delta_RhoSq": 0.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}
        wilson_coefficients = {"SM": 1.0,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}
        scale_factor = 1.0

        # Act: Create an instance
        cacher = modifier_hammer.HammerCacher(
            file_name, histo_name, ff_scheme, wilson_set, form_factors, wilson_coefficients, scale_factor
        )

        # Act & Assert: Case 1 - Cache matches input
        assert cacher.checkFFCache({"delta_RhoSq": 0.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}) is True

        # Act & Assert: Case 2 - Cache differs (new key added)
        assert cacher.checkFFCache({"delta_RhoSq": 1.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}) is False
        assert cacher._FFs == {"delta_RhoSq": 1.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}

    def test_getHistoTotalSM(self):
        # Arrange: Create an instance with basic setup
        # Arrange: Create an instance with pre-filled Wilson coefficients
        file_name = dir_path+"/hammer_file/hammer.dat"
        histo_name = "mmiss2_q2_el"
        ff_scheme = {"name":"SchemeBLPRXP","Process":"BtoD*","SchemeVar":"BLPRXPVar"}
        wilson_set = "BtoCTauNu"
        form_factors = {"delta_RhoSq": 0.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}
        wilson_coefficients = {"SM": 1.0,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}
        scale_factor = 1.0

        # Act: Create an instance
        cacher = modifier_hammer.HammerCacher(
            file_name, histo_name, ff_scheme, wilson_set, form_factors, wilson_coefficients, scale_factor
        )

        # Act: Compute total
        result = cacher.getHistoTotalSM()

        # Assert: Check the sum
        assert pytest.approx(result, 1e-0) == 93.  # Sum of the dummy weights

    def test_getHistoElementByPosNoScale(self):
        # Arrange: Create an instance with dummy data
        file_name = dir_path+"/hammer_file/hammer.dat"
        histo_name = "mmiss2_q2_el"
        ff_scheme = {"name":"SchemeBLPRXP","Process":"BtoD*","SchemeVar":"BLPRXP"}
        wilson_set = "BtoCTauNu"
        form_factors = {"delta_RhoSq": 0.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0}
        wilson_coefficients = {"SM": 1.0,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}
        scale_factor = 1.0

        # Act: Create an instance
        cacher = modifier_hammer.HammerCacher(
            file_name, histo_name, ff_scheme, wilson_set, form_factors, wilson_coefficients, scale_factor
        )

        # Act: Retrieve element
        result = cacher.getHistoElementByPosNoScale(35, {"SM": 1.0,"S_qLlL": 0.0,"S_qRlL": 0.0,"V_qLlL": 0.0,"V_qRlL": 0.0,"T_qLlL": 0.0}, {"delta_RhoSq": 0.0,"delta_cSt": 0.0,"delta_chi21": 0.0,"delta_chi2p": 0.0,"delta_chi3p": 0.0,"delta_eta1": 0.0,"delta_etap": 0.0,"delta_phi1p": 0.0,"delta_beta21": 0.0,"delta_beta3p": 0.0})

        # Assert: Check the result
        assert pytest.approx(result, 1e-2) == 3.76  # Element at position 20

    ### Test the MultiHammerCacher

class TestMultiHammerCacher:
    class DummyCacher:
        def __init__(self, scaleFactor, nobs, strides, wcs, ffs, norm_factor,element_values):
            self._scaleFactor = scaleFactor
            self._nobs = nobs
            self._strides = strides
            self._wcs = wcs
            self._FFs = ffs
            self._normFactor = norm_factor
            self.element_values = element_values

        def getHistoElementByPosNoScale(self, pos, wcs, FFs):
            return self.element_values[pos]

        def getHistoTotal(self):
            return self._normFactor

        def getHistoTotalSM(self):
            return self._normFactor

    def test_multihammer_constructor(self):

        cacher1 = TestMultiHammerCacher.DummyCacher(1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 100,[10,20,30])
        cacher2 = TestMultiHammerCacher.DummyCacher(1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 200,[15,25,35])

        # Act: Create MultiHammerCacher
        multi_cacher = modifier_hammer.MultiHammerCacher([cacher1, cacher2])

        # Assert: Check attributes
        assert multi_cacher._scaleFactor == 1.0
        assert multi_cacher._nobs == 3
        assert multi_cacher._strides == [1, 2, 3]
        assert multi_cacher._wcs == {"WC1": 1.0}
        assert multi_cacher._FFs == {"FF1": 1.0}
        assert multi_cacher._normFactor == 300
        assert len(multi_cacher._cacherList) == 2

    def test_multihammer_getHistoElementByPos(self):

        cacher1 = TestMultiHammerCacher.DummyCacher(1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 100,[10,20,30])
        cacher2 = TestMultiHammerCacher.DummyCacher(1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 200,[15,25,35])

        multi_cacher = modifier_hammer.MultiHammerCacher([cacher1, cacher2])
        wcs = {"WC1": 1.0}
        ffs = {"FF1": 1.0}

        # Act: Compute histogram element
        result = multi_cacher.getHistoElementByPos(1, wcs, ffs)  # Index 1

        # Assert: Check the result
        expected_res = ((20 + 25) * 1.0) / 300  # Scale factor (1.0) divided by norm factor (300)
        assert result == expected_res

    def test_multihammer_getHistoElementByPosSM(self):

        cacher1 = TestMultiHammerCacher.DummyCacher(1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 100,[10,20,30])
        cacher2 = TestMultiHammerCacher.DummyCacher(1.0, 3, [1, 2, 3], {"WC1": 1.0}, {"FF1": 1.0}, 200,[15,25,35])

        multi_cacher = modifier_hammer.MultiHammerCacher([cacher1, cacher2])
        wcs = {"WC1": 1.0}
        ffs = {"FF1": 1.0}

        # Act: Compute histogram element
        result = multi_cacher.getHistoElementByPosSM(1, wcs, ffs)  # Index 1

        # Assert: Check the result
        expected_res = ((20 + 25) * 1.0) / 300  # Scale factor (1.0) divided by norm factor (300)
        assert result == expected_res

class TestBackgroundCacher:
    @pytest.fixture(scope="class")
    def root_file_path(self):
        # Assuming the file and histogram already exist in this location
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, "hammer_file", "test_hammer_file.dat")
        return file_path

    def test_valid_histogram(self, root_file_path):
        # Arrange
        cacher = modifier_hammer.BackgroundCacher(root_file_path, "histo", [1, 2, 3])

        # Assert
        assert cacher._fileName == root_file_path
        assert cacher._histoName == "histo"
        assert cacher._strides == [1, 2, 3]
        assert np.array_equal(cacher._histo, np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        assert cacher._nobs == 10
        assert cacher._normFactor == 550  # Sum of [10, 20, ..., 100]

    def test_getHistoElementByPos(self, root_file_path):
        # Arrange
        cacher = modifier_hammer.BackgroundCacher(root_file_path, "histo", [1, 2, 3])

        # Act
        pos = 2  # Index 2 (third element in Python indexing)
        element = cacher.getHistoElementByPos(pos, {}, {})

        # Assert
        assert element == 30 / 550  # Normalize by total sum

##TEST HAMMERNUISWRAPPER, HAMMERNUISWRAPPERSM and BACKGROUNDNUISWRAPPER

class TestHammerNuisWrapper:

    @pytest.fixture
    def mock_hac(self):
        # Creating a mock or simple HammerCacher object for testing purposes
        # Simulating an example HammerCacher object with minimal data
        class MockHammerCacher:
            def __init__(self):
                self._nobs = 10
                self._wcs = {'SM': 1.0, 'S_qLlL': 1.0, 'S_qRlL': 1.0, 'V_qLlL': 1.0, 'V_qRlL': 1.0, 'T_qLlL': 1.0}
                self._FFs = {'FF1': 1.0, 'FF2': 1.0}
                self._strides = [1, 2, 3]

            def getHistoElementByPos(self, pos, wcs, FFs):
                # Simple mock function returning some calculated value based on pos
                return 100.0  # Return some dummy value for testing

        return MockHammerCacher()

    def test_initialization(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapper(mock_hac, param1=2, param2=3)

        # Assert
        assert wrapper._hac == mock_hac
        assert wrapper._nobs == 10
        assert wrapper._wcs == mock_hac._wcs
        assert wrapper._FFs == mock_hac._FFs
        assert wrapper._params == {'param1': 2, 'param2': 3}
        assert wrapper._strides == mock_hac._strides
        assert wrapper._dim == len(mock_hac._strides)
        assert wrapper._nbin == 0  # _nbin is initialized to 0

    def test_set_wcs(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapper(mock_hac, param1=2)

        # Act
        new_wcs = {
            'SM': 1.0,
            'Re_S_qLlL': 1.0, 'Im_S_qLlL': 2.0,
            'Re_S_qRlL': 1.0, 'Im_S_qRlL': 2.0,
            'Re_V_qLlL': 1.0, 'Im_V_qLlL': 2.0,
            'Re_V_qRlL': 1.0, 'Im_V_qRlL': 2.0,
            'Re_T_qLlL': 1.0, 'Im_T_qLlL': 2.0
        }
        wrapper.set_wcs(new_wcs)

        # Assert
        assert wrapper._wcs == {
            "SM": 1.0,
            "S_qLlL": complex(1.0, 2.0),
            "S_qRlL": complex(1.0, 2.0),
            "V_qLlL": complex(1.0, 2.0),
            "V_qRlL": complex(1.0, 2.0),
            "T_qLlL": complex(1.0, 2.0)
        }

    def test_set_FFs(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapper(mock_hac, param1=2, param2=3)

        # Act
        new_FFs = {'FF1': 3.0, 'FF2': 4.0}
        wrapper.set_FFs(new_FFs)

        # Assert
        assert wrapper._FFs == {'FF1': 3.0, 'FF2': 4.0}

    def test_set_params(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapper(mock_hac, param1=2, param2=3)

        # Act
        new_params = {'param1': 5, 'param2': 6}
        wrapper.set_params(new_params)

        # Assert
        assert wrapper._params == {'param1': 5, 'param2': 6}

    def test_set_nbin(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapper(mock_hac, param1=2, param2=3)

        # Act
        wrapper.set_nbin(5)

        # Assert
        assert wrapper._nbin == 5

    def test_evaluate(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapper(mock_hac, param1=2, param2=3)
        wrapper.set_nbin(1)
        wrapper.set_wcs({
            'SM': 1.0,
            'Re_S_qLlL': 1.0, 'Im_S_qLlL': 0.5,
            'Re_S_qRlL': 1.0, 'Im_S_qRlL': 0.5,
            'Re_V_qLlL': 1.0, 'Im_V_qLlL': 0.5,
            'Re_V_qRlL': 1.0, 'Im_V_qRlL': 0.5,
            'Re_T_qLlL': 1.0, 'Im_T_qLlL': 0.5
        })
        wrapper.set_FFs({'FF1': 1.0, 'FF2': 2.0})

        # Act
        result = wrapper.evaluate()

        # Assert
        # Since evaluate calls getHistoElementByPos(1) which returns 100.0,
        # and multiplies by the params, we expect a result of 100.0 * 1 (default) * 2
        assert result == 600.0  # The value returned from `getHistoElementByPos(1)` is 100.0

class TestHammerNuisWrapperSM:

    @pytest.fixture
    def mock_hac(self):
        # Creating a mock or simple HammerCacher object for testing purposes
        # Simulating an example HammerCacher object with minimal data
        class MockHammerCacher:
            def __init__(self):
                self._nobs = 10
                self._wcs = {'SM': 1.0, 'S_qLlL': 1.0, 'S_qRlL': 1.0, 'V_qLlL': 1.0, 'V_qRlL': 1.0, 'T_qLlL': 1.0}
                self._FFs = {'FF1': 1.0, 'FF2': 1.0}
                self._strides = [1, 2, 3]

            def getHistoElementByPosSM(self, pos, wcs, FFs):
                # Simple mock function returning some calculated value based on pos
                return 100.0  # Return some dummy value for testing

        return MockHammerCacher()

    def test_initialization(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapperSM(mock_hac, param1=2, param2=3)

        # Assert
        assert wrapper._hac == mock_hac
        assert wrapper._nobs == 10
        assert wrapper._wcs == mock_hac._wcs
        assert wrapper._FFs == mock_hac._FFs
        assert wrapper._params == {'param1': 2, 'param2': 3}
        assert wrapper._strides == mock_hac._strides
        assert wrapper._dim == len(mock_hac._strides)
        assert wrapper._nbin == 0  # _nbin is initialized to 0

    def test_set_wcs(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapperSM(mock_hac, param1=2)

        # Act
        new_wcs = {
            'SM': 1.0,
            'Re_S_qLlL': 1.0, 'Im_S_qLlL': 0.5,
            'Re_S_qRlL': 1.0, 'Im_S_qRlL': 0.5,
            'Re_V_qLlL': 1.0, 'Im_V_qLlL': 0.5,
            'Re_V_qRlL': 1.0, 'Im_V_qRlL': 0.5,
            'Re_T_qLlL': 1.0, 'Im_T_qLlL': 0.5
        }
        wrapper.set_wcs(new_wcs)

        # Assert
        assert wrapper._wcs == {
            "SM": 1.0,
            "S_qLlL": complex(1.0, 0.5),
            "S_qRlL": complex(1.0, 0.5),
            "V_qLlL": complex(1.0, 0.5),
            "V_qRlL": complex(1.0, 0.5),
            "T_qLlL": complex(1.0, 0.5)
        }

    def test_set_FFs(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapperSM(mock_hac, param1=2, param2=3)

        # Act
        new_FFs = {'FF1': 3.0, 'FF2': 4.0}
        wrapper.set_FFs(new_FFs)

        # Assert
        assert wrapper._FFs == {'FF1': 3.0, 'FF2': 4.0}

    def test_set_params(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapperSM(mock_hac, param1=2, param2=3)

        # Act
        new_params = {'param1': 5, 'param2': 6}
        wrapper.set_params(new_params)

        # Assert
        assert wrapper._params == {'param1': 5, 'param2': 6}

    def test_set_nbin(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapperSM(mock_hac, param1=2, param2=3)

        # Act
        wrapper.set_nbin(5)

        # Assert
        assert wrapper._nbin == 5

    def test_evaluate(self, mock_hac):
        # Arrange
        wrapper = modifier_hammer.HammerNuisWrapperSM(mock_hac, param1=2, param2=3)
        wrapper.set_nbin(1)
        wrapper.set_wcs({
            'SM': 1.0,
            'S_qLlL': 1.0, 'S_qLlL_im': 0.5,
            'S_qRlL': 1.0, 'S_qRlL_im': 0.5,
            'V_qLlL': 1.0, 'V_qLlL_im': 0.5,
            'V_qRlL': 1.0, 'V_qRlL_im': 0.5,
            'T_qLlL': 1.0, 'T_qLlL_im': 0.5
        })
        wrapper.set_FFs({'FF1': 1.0, 'FF2': 2.0})

        # Act
        result = wrapper.evaluate()

        # Assert
        # Since evaluate calls getHistoElementByPosSM(1) which returns 100.0,
        # and multiplies by the params, we expect a result of 100.0 * 1 (default) * 2
        assert result == 600.0  # The value returned from `getHistoElementByPosSM(1)` is 100.0

class TestBackgroundNuisWrapper:

    @pytest.fixture
    def mock_bkg(self):
        # Creating a mock BackgroundCacher object for testing purposes
        class MockBackgroundCacher:
            def __init__(self):
                self._nobs = 10
                self._wcs = {'SM': 1.0, 'S_qLlL': 1.0, 'S_qRlL': 1.0, 'V_qLlL': 1.0, 'V_qRlL': 1.0, 'T_qLlL': 1.0}
                self._FFs = {'FF1': 1.0, 'FF2': 1.0}
                self._strides = [1, 2, 3]

            def getHistoElementByPos(self, pos, wcs, FFs):
                # Simple mock function returning some calculated value based on pos
                return 100.0  # Return some dummy value for testing

        return MockBackgroundCacher()

    def test_initialization(self, mock_bkg):
        # Arrange
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        # Assert
        assert wrapper._bkg == mock_bkg
        assert wrapper._nobs == 10
        assert wrapper._wcs == {}
        assert wrapper._FFs == {}
        assert wrapper._params == {'param1': 2, 'param2': 3}
        assert wrapper._strides == mock_bkg._strides
        assert wrapper._dim == len(mock_bkg._strides)
        assert wrapper._nbin == 0  # _nbin is initialized to 0

    def test_set_nbin(self, mock_bkg):
        # Arrange
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        # Act
        wrapper.set_nbin(5)

        # Assert
        assert wrapper._nbin == 5

    def test_set_wcs(self, mock_bkg):
        # Arrange
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        # Act
        wrapper.set_wcs({'SM': 1.0, 'S_qLlL': 1.0})

        # Assert
        assert wrapper._wcs == {}

    def test_set_FFs(self, mock_bkg):
        # Arrange
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        # Act
        wrapper.set_FFs({'FF1': 3.0, 'FF2': 4.0})

        # Assert
        assert wrapper._FFs == {}

    def test_set_params(self, mock_bkg):
        # Arrange
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)

        # Act
        new_params = {'param1': 5, 'param2': 6}
        wrapper.set_params(new_params)

        # Assert
        assert wrapper._params == {'param1': 5, 'param2': 6}

    def test_evaluate(self, mock_bkg):
        # Arrange
        wrapper = modifier_hammer.BackgroundNuisWrapper(mock_bkg, param1=2, param2=3)
        wrapper.set_nbin(1)
        wrapper.set_wcs({'SM': 1.0, 'S_qLlL': 1.0})
        wrapper.set_FFs({'FF1': 1.0, 'FF2': 2.0})

        # Act
        result = wrapper.evaluate()

        # Assert
        # Since evaluate calls getHistoElementByPos(1) which returns 100.0,
        # and multiplies by the params, we expect a result of 100.0 * 2 (default)
        assert result == 600.0  # The value returned from `getHistoElementByPos(1)` is 100.0

### TEST TEMPLATE

class TestTemplateClass:

    @pytest.fixture
    def mock_wrapper(self):
        # Creating a mock BackgroundNuisWrapper object for testing purposes
        class MockWrapper:
            def __init__(self):
                self._nobs = 10
                self._wcs = {'SM': 1.0, 'S_qLlL': 1.0, 'S_qRlL': 1.0, 'V_qLlL': 1.0, 'V_qRlL': 1.0, 'T_qLlL': 1.0}
                self._FFs = {'FF1': 1.0, 'FF2': 1.0}
                self._params = {'param1': 1.0, 'param2': 2.0}
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
                # A simple mock evaluation function
                return 1.0*self._wcs['SM']  # Return a fixed value for testing

        return MockWrapper()

    def test_initialization(self, mock_wrapper):
        # Arrange
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        # Assert
        assert obj._name == "TestTemplate"
        assert obj._wrap == mock_wrapper
        assert obj._nobs == 10
        assert obj._nwcs == len(mock_wrapper._wcs)
        assert obj._nFFs == len(mock_wrapper._FFs)
        assert obj._nparams == len(mock_wrapper._params)
        assert obj._strides == mock_wrapper._strides

    def test_generate_template(self, mock_wrapper):
        # Arrange
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        # Act
        bin_contents = obj.generate_template(SM=1.0, S_qLlL=1.1, S_qRlL=1.2, V_qLlL=1.3, V_qRlL=1.4, T_qLlL=1.5,
                                             FF1=2.0, FF2=3.0, param1=4.0, param2=5.0)

        # Assert
        assert len(bin_contents) == 10  # Should match the number of observations
        assert np.all(bin_contents == 1.0)  # Each bin should have the same value, since evaluate() returns 1.0

    def test_generate_toy(self, mock_wrapper):
        # Arrange
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        # Act
        bin_contents = obj.generate_toy(SM=1.0, S_qLlL=1.1, S_qRlL=1.2, V_qLlL=1.3, V_qRlL=1.4, T_qLlL=1.5,
                                        FF1=2.0, FF2=3.0, param1=4.0, param2=5.0)

        # Assert
        assert len(bin_contents) == 10  # Should match the number of observations
        assert np.all(bin_contents >= 0)  # Since it's Poisson, all values should be non-negative
        assert isinstance(bin_contents[0], np.float64)  # Each bin value should be a float type
        assert np.any(bin_contents != 10.0)  # The values should vary due to the Poisson distribution

    def test_generate_template_with_different_params(self, mock_wrapper):
        # Arrange
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        # Act
        bin_contents = obj.generate_template(SM=2.0, S_qLlL=1.0, S_qRlL=1.0, V_qLlL=1.0, V_qRlL=1.0, T_qLlL=1.0,
                                             FF1=3.0, FF2=4.0, param1=2.0, param2=3.0)

        # Assert
        assert np.all(bin_contents == 2.0)  # Evaluate should return 1.0*SM
        assert len(bin_contents) == 10  # Length should match the number of observations (10)

    def test_generate_toy_with_different_params(self, mock_wrapper):
        # Arrange
        obj = modifier_hammer.template("TestTemplate", mock_wrapper)

        # Act
        bin_contents = obj.generate_toy(SM=2.0, S_qLlL=1.0, S_qRlL=1.0, V_qLlL=1.0, V_qRlL=1.0, T_qLlL=1.0,
                                        FF1=3.0, FF2=4.0, param1=2.0, param2=3.0)

        # Assert
        assert len(bin_contents) == 10  # Length should match the number of observations (10)
        assert np.all(bin_contents >= 0)  # Poisson distribution guarantees non-negative values
        assert isinstance(bin_contents[0], np.float64)  # The result should be a float type

### TEST FITTER

class TestFitterClass:

    @pytest.fixture
    def mock_template(self):
        # Creating a mock template object for testing purposes
        class MockTemplate:
            def __init__(self):
                self._nobs = 10  # Assuming the template has 10 observations

            def generate_template(self, **kwargs):
                return np.ones(self._nobs) * 10  # Returns a simple array with all values = 10

        return MockTemplate()

    def test_initialization(self, mock_template):
        # Arrange
        obj = modifier_hammer.fitter([mock_template])

        # Assert
        assert obj._template_list == [mock_template]  # Template list should contain one mock template
        assert np.array_equal(obj._data, np.array([]))  # Data should be an empty array initially

    def test_get_template(self, mock_template):
        # Arrange
        obj = modifier_hammer.fitter([mock_template])

        # Act
        template = obj.get_template(0)

        # Assert
        assert template == mock_template  # Should return the first template in the list

    def test_upload_data(self, mock_template):
        # Arrange
        obj = modifier_hammer.fitter([mock_template])
        data = np.array([1, 2, 3, 4, 5])

        # Act
        obj.upload_data(data)

        # Assert
        assert np.array_equal(obj._data, data)  # Data should be equal to the uploaded data
        assert obj._data.shape == data.shape  # Data shape should match the shape of the input data

    def test_generate_template_integration(self, mock_template):
        # Arrange
        obj = modifier_hammer.fitter([mock_template])

        # Act
        obj.upload_data(np.array([1, 2, 3, 4, 5]))  # Upload data for the fitter
        generated_template = obj.get_template(0).generate_template(SM=1.0)  # Generate template with mock template

        # Assert
        assert len(generated_template) == 10  # Template should have 10 observations (based on mock template)
        assert np.all(generated_template == 10)  # All values in the template should be 10
