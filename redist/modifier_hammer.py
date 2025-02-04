from hammer.hammerlib import (Hammer, IOBuffer,
                              RecordType)

from copy import deepcopy
import numpy as np
import json
import pyhf

from redist.modifier import Modifier

from redist import modifier

class Modifier_Hammer(Modifier):
    def __init__(self, new_pars, alt_dist, null_dist, name = None, cutoff=None, weight_bound=None, allow_negative_weights=False):
        """
        Args:
            new_pars (dict): New parameters to parametrize the model.
            alt_dist (callable): Alternative distribution to be tested.
            null_dist (callable): Null distribution of the nominal model.
            name (string, optional): Name of the custom modifier. Defaults to None.
            cutoff (tuple, optional): Kinematic cutoff values to limit the integration boundaries to a given range. Defaults to None.
            weight_bound (float, optional): Upper bound on the weight. Defaults to None.
            allow_negative_weights (bool, optional): Allow negative weights. Defaults to False.
        """
        # store name and cutoff
        self.name = name if name else 'custom'
        self.cutoff = cutoff
        self.weight_bound = weight_bound
        self.allow_negative_weights = allow_negative_weights

        # store null and alternative distributions
        self.null_dist = null_dist
        self.alt_dist  = alt_dist

        # compute the bin-integrated null distribution (this is fixed)
        self.null_binned = null_dist()

        # take care of correlated paramters
        self.new_pars = new_pars
        self.corr_pars, self.unco_pars = self._separate_pars(new_pars)
        self.corr_infos = self._corr_infos(self.corr_pars)

        # cache previously called function values
        self.cache = {}

    def get_weights(self, pars):
        """
        Compute the new weights and process them for sensibility.

        Args:
            pars (dict): pyhf parameters.

        Returns:
            array: Weights for the given parameters.
        """
        # compute original parameters from pyhf parameters
        rot_pars = self.rotate_pars(pars)
        alt_binned = self.alt_dist(**rot_pars)#bintegrate(self.alt_dist, self.bins, tuple(rot_pars.values()), cutoff=self.cutoff)
        weights = np.array(alt_binned) / np.array(self.null_binned)

        weights[np.isnan(weights)] = 1.
        if not self.allow_negative_weights:
            weights[weights<0.] = 1.
        if self.weight_bound:
            weights[weights>self.weight_bound] = self.weight_bound

        #flatten the weights
        weights = weights.reshape(-1, order='F')
        return weights

    def weight_func(self, pars):
        """
        Build function that applies weights to histogram.

        Args:
            pars (dict): pyhf parameters.

        Returns:
            callable: Function that returns histogram modifications.
        """
        key = tuple(i for i in pars.items())
        if key in self.cache:
            return self.cache[key]

        weights = self.get_weights(pars)
        results = weights

        def func():
            return results

        self.cache[key] = func

        return func

def save_hammer(file, spec, cmods, data=None):
    """
    Save the custom model, mapping distribution (and data).

    Args:
        file (string): File name.
        spec (dict): Model specification.
        cmods (list): List of custom modifiers.
        data (array, optional): Data to be saved. Defaults to None.
    """
    d = {
        'spec'          : spec,
        'name'          : [cmod.name for cmod in cmods],
        'new_pars'      : [cmod.new_pars for cmod in cmods],
        'cutoff'        : [cmod.cutoff for cmod in cmods],
        'weight_bound'  : [cmod.weight_bound for cmod in cmods]
        }
    if data is not None:
        d['data'] = np.array(data).tolist()

    with open(file, 'w') as f:
        json.dump(d, f, indent=4)

def load_hammer(file, alt_dist, null_dist, return_modifier=False, return_data=False, **kwargs):
    """
    Load and build model from file

    Args:
        file (string): File name.
        alt_dist (callable): Alternative distribution to be tested.
        null_dist (callable): Null distribution of the nominal model.
        return_modifier (bool, optional): Return custom modifiers. Defaults to False.
        return_data (bool, optional): Return data. Defaults to False.
        kwargs: Additional arguments for the pyhf model.

    Returns:
        pyhf.Model, list, array: Model, custom modifiers, data.
    """
    with open(file, 'r') as f:
        d = json.load(f)

    new_pars = {}
    for pars in d['new_pars']:
        new_pars.update(modifier._read_pars(pars))
    cmods = []
    for name, cutoff, weight_bound in zip(d['name'], d['cutoff'], d['weight_bound']):
        cmods.append(Modifier_Hammer(new_pars, alt_dist, null_dist,
                              name=name, cutoff=cutoff, weight_bound=weight_bound))

    expanded_pyhf = {}
    for cmod in cmods:
        expanded_pyhf.update(cmod.expanded_pyhf)

    model = pyhf.Model(d['spec'], validate=False, batch_size=None, modifier_set=expanded_pyhf, **kwargs)

    if return_modifier and return_data: return model, cmods, d['data']
    if return_modifier: return model, cmods
    if return_data: return model, d['data']
    return model

# the hammer cacher class handles directly the hammer histogram
# it access it and it changes, if required, the FF and the WC d.o.f
# giving access to the histogram as it changes wrt them
class HammerCacher:
    def __init__(self, fileName, histoName, FFscheme, WilsonSet, FormFactors, WilsonCoefficients, scaleFactor, verbose=False):#, **kwargs):
        self._histoName = histoName
        self._FFScheme = FFscheme
        self._WilsonSet = WilsonSet
        self._scaleFactor = scaleFactor

        self._wcs = WilsonCoefficients
        self._FFs = FormFactors

        self._nobs = 1
        self._strides = [1]
        self._ham = Hammer()
        self._ham.set_units("GeV")

        buf = IOBuffer(RecordType.UNDEFINED)
        if(verbose):
            print(f"fileName = {fileName}")
            print(f"histoName = {histoName}")

        with open(fileName, 'rb', buffering=0) as fin:
            if buf.load(fin) and self._ham.load_run_header(buf):
                self._ham.init_run()
                if buf.load(fin):
                    while buf.kind == RecordType.HISTOGRAM or buf.kind == RecordType.HISTOGRAM_DEFINITION:
                        if buf.kind == RecordType.HISTOGRAM_DEFINITION:
                            name = self._ham.load_histogram_definition(buf)
                        else:
                            info = self._ham.load_histogram(buf)
                        if not buf.load(fin):
                            break

        self._ham.set_ff_eigenvectors(self._FFScheme["Process"],self._FFScheme["SchemeVar"],self._FFs)
        self._ham.set_wilson_coefficients(self._WilsonSet, self._wcs)
        self._histo = self._ham.get_histogram(histoName, FFscheme["name"])
        dims = self._ham.get_histogram_shape(histoName)
        dims = dims[1:] + dims[:1]
        dims.pop()

        ndims = self._ham.get_histogram_shape(histoName)

        for ndim in ndims:
            self._nobs*=ndim
        for dim in dims:
            self._strides = [c * dim for c in self._strides]
            self._strides.append(1)

        self._normFactor = self.getHistoTotalSM()

    def checkWCCache(self, wcs):
        isCached = True
        for key in wcs:
            if key not in self._wcs.keys():
                self._wcs[key] = wcs[key]
                isCached = False
            elif not (self._wcs[key] - wcs[key])==0:
                self._wcs[key] = wcs[key]
                isCached = False
        return isCached

    def checkFFCache(self, FFs):
        isCached = True
        for key in FFs:
            if key not in self._FFs.keys():
                self._FFs[key] = FFs[key]
                isCached = False
            elif not (self._FFs[key] - FFs[key])==0:
                self._FFs[key] = FFs[key]
                isCached = False
        return isCached

    def getHistoTotalSM(self):
        total = 0
        wcs = {}
        for key, value in self._wcs.items():
            if key == 'SM':
                wcs[key] = 1.
            else:
                wcs[key] = 0.
        self._ham.reset_wilson_coefficients(self._WilsonSet)
        self._ham.set_wilson_coefficients(self._WilsonSet, wcs)
        self._histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        for ni in range(self._nobs):
            total += self._histo[ni].sum_wi
        return total

    def getHistoElementByPosNoScale(self, pos, wcs, FFs):
        if not self.checkFFCache(FFs):
            self._ham.reset_ff_eigenvectors(self._FFScheme["Process"], self._FFScheme["SchemeVar"])
            self._ham.set_ff_eigenvectors(self._FFScheme["Process"], self._FFScheme["SchemeVar"], FFs)
            self._histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        if not self.checkWCCache(wcs):
            self._ham.reset_wilson_coefficients(self._WilsonSet)
            self._ham.set_wilson_coefficients(self._WilsonSet, wcs)
            self._histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        return self._histo[pos].sum_wi

    def getHistoElementByPosNoScaleSM(self, pos, wcs, FFs):
        for key in wcs.keys():
            if key != 'SM':
                wcs[key] = 0.
        if not self.checkFFCache(FFs):
            self._ham.set_ff_eigenvectors(self._FFScheme["Process"], self._FFScheme["SchemeVar"], FFs)
            self._histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        if not self.checkWCCache(wcs):
            self._ham.reset_wilson_coefficients(self._WilsonSet)
            self._ham.set_wilson_coefficients(self._WilsonSet, wcs)
            self._histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        return self._histo[pos].sum_wi

# Multi hammer cacher allows you to store multiple histograms in multiple files
# and treat them like a single one for when you parallelize the hammer reweighting process
# that can be very time consuming
class MultiHammerCacher:
    def __init__(self, cacherList):
        cacher0 = cacherList[0]
        self._cacherList = []
        self._normFactor = 0
        self._scaleFactor = cacher0._scaleFactor
        self._nobs = cacher0._nobs
        self._strides = cacher0._strides
        self._wcs = cacher0._wcs
        self._FFs = cacher0._FFs
        for cacher in cacherList:
            self._cacherList.append(cacher)
            self._normFactor += cacher.getHistoTotalSM()

    def getHistoElementByPos(self, pos, wcs, FFs):
        res = 0
        for i in range(len(self._cacherList)):
            res += self._cacherList[i].getHistoElementByPosNoScale(pos,wcs,FFs)
        self._wcs = wcs
        self._FFs = FFs
        return res * self._scaleFactor / self._normFactor

    def getHistoElementByPosSM(self, pos, wcs, FFs):
        res = 0
        for key in wcs.keys():
            if key != 'SM':
                wcs[key] = 0.
        for i in range(len(self._cacherList)):
            res += self._cacherList[i].getHistoElementByPosNoScale(pos,wcs,FFs)
        self._wcs = wcs
        self._FFs = FFs
        return res * self._scaleFactor / self._normFactor

# the background cacher access not hammer reweighted histograms and gives us in a format
# similar to the HammerCacher (easier to handle them together later)
class BackgroundCacher:
    def __init__(self, fileName, histoName, strides):
        self._fileName = fileName
        self._histoName = histoName
        self._strides = strides
        try:
            self._histo = np.loadtxt(self._fileName)
        except Exception as e:
            print(f"Error: Could not read file '{self._fileName}'. Exception: {e}")
            return
        if len(self._histo) == 0:
            print(f"Error: Histogram data in file '{self._fileName}' is empty.")
            return
        self._nobs = len(self._histo)
        self._normFactor = self._histo.sum()

    def getHistoElementByPos(self, pos,wcs,FFs):
        return self._histo[pos] / self._normFactor


# here we define the multiplicative nuisance parameters to apply to the hammer
# reweighted histogram
# the wrapper can change the d.o.f of the contribution and returns the content of
# a given bin wrt to the current values of the d.o.f. with an evaluate function
class HammerNuisWrapper:
    def __init__(self, hac, **kwargs):
        self._hac = hac
        self._nobs = hac._nobs
        self._wcs = hac._wcs
        self._FFs = hac._FFs
        self._params = {}
        for key, value in kwargs.items():
            self._params[key] = value
        self._nbin = 0
        self._strides = hac._strides
        self._dim = len(hac._strides)

    def set_wcs(self,wcs):
        self._wcs = {"SM":wcs[list(wcs.keys())[0]],"S_qLlL": complex(wcs[list(wcs.keys())[1]], wcs[list(wcs.keys())[2]]),"S_qRlL": complex(wcs[list(wcs.keys())[3]], wcs[list(wcs.keys())[4]]),"V_qLlL": complex(wcs[list(wcs.keys())[5]], wcs[list(wcs.keys())[6]]),"V_qRlL": complex(wcs[list(wcs.keys())[7]], wcs[list(wcs.keys())[8]]),"T_qLlL": complex(wcs[list(wcs.keys())[9]], wcs[list(wcs.keys())[10]])}

    def set_FFs(self,FFs):
        FFs_temp = {}
        for key, value in FFs.items():
            if key in self._FFs.keys():
                FFs_temp[key] = float(value)
        self._FFs = FFs_temp

    def set_params(self,params):
        params_temp = {}
        for key, value in params.items():
            if key in self._params.keys():
                params_temp[key] = value
        self._params = params_temp

    def set_nbin(self,nbin):
        self._nbin = nbin

    def evaluate(self):
        val = self._hac.getHistoElementByPos(self._nbin, self._wcs, self._FFs)
        for key, value in self._params.items():
            val = val*value
        return val

# this is the same as HammerNuisWrapper but the evaluate method is
# ignoring the WCs
# if you for example want to inject new physics B2DTauNu and not in B2DMuNu
# you'll build a ordinary HammerNuisWrapper for B2DTauNu
# and a SM one for B2DMuNu
class HammerNuisWrapperSM:
    def __init__(self, hac, **kwargs):
        self._hac = hac
        self._nobs = hac._nobs
        self._wcs = hac._wcs
        self._FFs = hac._FFs
        self._params = {}
        for key, value in kwargs.items():
            self._params[key] = value
        self._nbin = 0
        self._strides = hac._strides
        self._dim = len(hac._strides)

    def set_wcs(self,wcs):
        #for key in self._wcs.keys():
        #    if key == 'SM':
        #        self._wcs[key] = wcs[key]
        #    else:
        #        self._wcs[key] = complex(wcs['Re_'+key],wcs['Im_'+key])
        self._wcs = {"SM":wcs[list(wcs.keys())[0]],"S_qLlL": complex(wcs[list(wcs.keys())[1]], wcs[list(wcs.keys())[2]]),"S_qRlL": complex(wcs[list(wcs.keys())[3]], wcs[list(wcs.keys())[4]]),"V_qLlL": complex(wcs[list(wcs.keys())[5]], wcs[list(wcs.keys())[6]]),"V_qRlL": complex(wcs[list(wcs.keys())[7]], wcs[list(wcs.keys())[8]]),"T_qLlL": complex(wcs[list(wcs.keys())[9]], wcs[list(wcs.keys())[10]])}

    def set_FFs(self,FFs):
        FFs_temp = {}
        for key, value in FFs.items():
            if key in self._FFs.keys():
                FFs_temp[key] = float(value)
        self._FFs = FFs_temp

    def set_params(self,params):
        params_temp = {}
        for key, value in params.items():
            if key in self._params.keys():
                params_temp[key] = value
        self._params = params_temp

    def set_nbin(self,nbin):
        self._nbin = nbin

    def evaluate(self):
        val = self._hac.getHistoElementByPosSM(self._nbin, self._wcs, self._FFs)
        for key, value in self._params.items():
            val = val*value
        return val

# this attaches Nuisance parameters to the BackgroundCacher
# notice that one of the Nuisance parameters here should always be the yield
class BackgroundNuisWrapper:
    def __init__(self, bkg, **kwargs):
        self._bkg = bkg
        self._nobs = bkg._nobs
        self._params = {}
        self._wcs={}
        self._FFs={}
        for key, value in kwargs.items():
            self._params[key] = value
        self._nbin = 0
        self._strides = bkg._strides
        self._dim = len(bkg._strides)


    def set_nbin(self,nbin):
        self._nbin = nbin

    def set_wcs(self,wcs):
        self._wcs = {}

    def set_FFs(self,FFs):
        self._FFs = {}

    def set_params(self,params):
        params_temp = {}
        for key, value in params.items():
            if key in self._params.keys():
                params_temp[key] = value
        self._params = params_temp

    def evaluate(self):
        val = self._bkg.getHistoElementByPos(self._nbin, self._wcs, self._FFs)
        for key, value in self._params.items():
            val = val*value
        return val

# the template class takes the wrapper and allows to generate templates, and toys
# wrt any set of d.o.f we want
class template:
    def __init__(self, name, wrap):
        self._name = name
        self._wrap = wrap
        self._nobs = wrap._nobs
        self._nwcs = len(self._wrap._wcs)
        self._nFFs = len(self._wrap._FFs)
        self._nparams = len(self._wrap._params)
        self._strides = wrap._strides

    def generate_template(self, **kwargs):
        wcs = {}
        FFs = {}
        params = {}

        for i, (key, value) in enumerate(kwargs.items()):
            if i < self._nwcs*2-1:
                wcs[key] = value
            elif self._nwcs*2-1 <= i < self._nwcs*2-1+self._nFFs:
                FFs[key] = value
            else:
                params[key] = value
        self._wrap.set_wcs(wcs)
        self._wrap.set_FFs(FFs)
        self._wrap.set_params(params)

        bin_contents = np.zeros(self._nobs)

        for i in range(self._nobs):
            self._wrap.set_nbin(i)
            val=self._wrap.evaluate()
            bin_contents[i]+=val

        return bin_contents

    def generate_toy(self, **kwargs):
        wcs = {}
        FFs = {}
        params = {}

        for i, (key, value) in enumerate(kwargs.items()):
            if i < self._nwcs*2-1:
                wcs[key] = value
            elif self._nwcs*2-1 <= i < self._nwcs*2-1+self._nFFs:
                FFs[key] = value
            else:
                params[key] = value
        self._wrap.set_wcs(wcs)
        self._wrap.set_FFs(FFs)
        self._wrap.set_params(params)

        bin_contents = np.zeros(self._nobs)

        for i in range(self._nobs):
            self._wrap.set_nbin(i)
            val=self._wrap.evaluate()
            bin_contents[i]+=np.random.poisson(val)

        return bin_contents

# the fitter contains a template list and data (toys in the examples)
# it contains the definition of a nul_pdf and an alternative_pdf to be injected in the definition of the modifier
# a small plotting interface is implemented to retireve the projected histograms (from the strides) and overlay data
class fitter:
    def __init__(self,template_list):
        self._template_list = template_list
        self._data = np.array([])

    def get_template(self,index):
        return self._template_list[index]

    def upload_data(self,data):
        self._data = data

# The reader class aim is to make everything above not necessary to be fully undestood
# A config file is provided and the reader produces itself the necessary objects:
# Cachers -> Wrappers -> Templates -> Fitter (returned)
# giving access to a fitter with a toy stored inside as data (temporary)
class Reader:
    def __init__(self, filename):
        self.name = filename
        with open(filename, 'r') as f:
            self.config = json.load(f)

    def createFitter(self, verbose=False):
        template_list = []

        for mode, mode_config in self.config.items():
            hac_list = []
            if verbose:
                print(f"Reading the mode: {mode}")
            fileNames = mode_config["fileNames"]
            histoname = mode_config["histoname"]
            ffscheme = mode_config["ffscheme"]
            wcscheme = mode_config["wcscheme"]
            formfactors = mode_config["formfactors"]
            wilsoncoefficients = mode_config["wilsoncoefficients"]
            scalefactor = mode_config["scalefactor"]
            nuisance = mode_config["nuisance"]
            is_hammer_weighted = mode_config["ishammerweighted"]
            injectNP = mode_config["injectNP"]
            strides = mode_config["strides"]
            _wilsoncoefficients = {}
            for key, value in wilsoncoefficients.items():
                _wilsoncoefficients[key] = complex(value[0],value[1])
            if is_hammer_weighted:
                for fileName in fileNames:
                    if verbose:
                        print(f"Reading {fileName}")
                    hac_list.append(HammerCacher(fileName, histoname, ffscheme, wcscheme, deepcopy(formfactors), deepcopy(_wilsoncoefficients), deepcopy(scalefactor)))
                cacher = MultiHammerCacher(hac_list)
                if injectNP:
                    wrapper = HammerNuisWrapper(cacher, **nuisance)
                    temp = template(mode, wrapper)
                    template_list.append(temp)
                else:
                    wrapper = HammerNuisWrapperSM(cacher, **nuisance)
                    temp = template(mode, wrapper)
                    template_list.append(temp)
            else:
                for fileName in fileNames:
                    if verbose:
                        print(f"Reading {fileName}")
                    hac_list.append(BackgroundCacher(fileName, histoname,strides))
                cacher = hac_list[0]
                wrapper = BackgroundNuisWrapper(cacher,**nuisance)
                temp = template(mode, wrapper)
                template_list.append(temp)

        return fitter(template_list)
