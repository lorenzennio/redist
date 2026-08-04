"""
Reweighting driven by HAMMER histogram files.

The rest of `redist` builds its weights by integrating a null and an alternative
distribution over a kinematic binning. HAMMER instead hands over templates that
are already binned in the analysis observables, reweighted for a choice of form
factors and Wilson coefficients, so there is no integration left to do: the
weight in a bin is simply the ratio of the two templates there.

HAMMER itself is optional and is deliberately not a dependency of `redist`. It
is not on PyPI -- it is a C++ library with a Cython wrapper, built from source --
so `redist/__init__.py` does not import this module, and this module does not
import HAMMER until a `HammerCacher` actually needs it. Everything else here,
including `Modifier_Hammer` and the background templates, works without it.

See the HAMMER section of the README for the build recipe.
"""

import json
import warnings
from copy import deepcopy

import numpy as np
import pyhf
from pyhf import get_backend

from redist import modifier
from redist.modifier import Modifier


def _hammerlib():
    """
    The HAMMER Python bindings, imported on demand.

    Importing at module scope would make `from redist import modifier_hammer`
    fail for everyone who has not built HAMMER, which is most users, and would
    turn an optional feature into a hard dependency.

    Returns:
        tuple: The `Hammer`, `IOBuffer` and `RecordType` bindings.

    Raises:
        ImportError: If HAMMER is not installed.
    """
    try:
        from hammer.hammerlib import Hammer, IOBuffer, RecordType
    except ImportError as exc:
        raise ImportError(
            "HAMMER is needed to read hammer histogram files and is not "
            "installed. It is not on PyPI and has to be built from source; see "
            "the HAMMER section of the redist README for the recipe."
        ) from exc
    return Hammer, IOBuffer, RecordType


def _sm_only(wcs):
    """
    A copy of `wcs` with every coefficient but the standard model one switched off.

    A copy, and not an edit in place. The dictionary handed in belongs to a
    wrapper that goes on using it, and often to several templates at once, so
    zeroing it in place -- as this once did -- silently switched off the
    coefficients of every other template sharing it.

    Args:
        wcs (dict): Wilson coefficients.

    Returns:
        dict: The same coefficients, with the non-SM ones set to zero.
    """
    return {key: (value if key == "SM" else 0.0) for key, value in wcs.items()}


class Modifier_Hammer(Modifier):
    """
    Modifier reweighting a histogram by the ratio of two HAMMER templates.

    Both distributions return a template already binned in the analysis
    observables, so, unlike `Modifier`, there is no kinematic binning, no
    mapping distribution and no quadrature: the weight of a bin is the ratio of
    the alternative to the null template in that bin, applied to the sample
    directly.

    HAMMER is a compiled library called outside the tensor graph, so this only
    runs on the numpy backend.
    """

    def __init__(
        self,
        new_pars,
        alt_dist,
        null_dist,
        name=None,
        cutoff=None,
        weight_bound=None,
        allow_negative_weights=False,
        cache_size=128,
    ):
        """
        Args:
            new_pars (dict): New parameters to parametrize the model.
            alt_dist (callable): Alternative distribution to be tested. Returns
                a template binned in the analysis observables.
            null_dist (callable): Null distribution of the nominal model, in the
                same binning, taking no parameters.
            name (string, optional): Name of the custom modifier. Defaults to None.
            cutoff (tuple, optional): Carried through `save_hammer` and
                `load_hammer` for compatibility with `Modifier`, and otherwise
                unused: a HAMMER template arrives already binned, so there are
                no integration boundaries to restrict. Defaults to None.
            weight_bound (float, optional): Upper bound on the weight. Weights
                above it are clipped to it. Any value is honoured, including
                zero and negatives, which are only meaningful together with
                `allow_negative_weights`. Defaults to None, meaning unbounded.
            allow_negative_weights (bool, optional): Allow negative weights.
                Defaults to False.
            cache_size (int, optional): How many parameter points to keep
                weights for. The cache is there for the repeated calls a single
                likelihood evaluation makes with identical parameters; a scan or
                a Markov chain never revisits a point, so without a bound the
                cache would only grow. Least recently used entries are dropped
                first, and zero disables it. Cannot change any result, so it is
                not saved with the model. Defaults to 128.
        """
        # store name and cutoff
        self.name = name if name else "custom"
        self.cutoff = cutoff
        self.weight_bound = weight_bound
        self.allow_negative_weights = allow_negative_weights

        # store null and alternative distributions
        self.null_dist = null_dist
        self.alt_dist = alt_dist

        # the null template is fixed, so it is computed once. It is kept exactly
        # as the distribution returned it, since it is public, and converted
        # only for the arrays derived from it below.
        self.null_binned = null_dist()
        null_binned = np.asarray(self.null_binned, dtype=float)

        # A bin the null template does not populate carries no ratio. Unlike the
        # quadrature path in `Modifier`, which refuses to build at all, an empty
        # bin is ordinary here: a HAMMER template is a sparse multi-dimensional
        # histogram and its corners are routinely empty. Those bins are left
        # unweighted, and counted out loud so that a badly chosen binning is
        # visible rather than silent.
        self._invalid = ~(np.isfinite(null_binned) & (null_binned != 0.0))
        if self._invalid.any():
            warnings.warn(
                f"the null template is empty or not finite in "
                f"{int(self._invalid.sum())} of {self._invalid.size} bins, which "
                "therefore carry no reweighting ratio and are left at a weight "
                "of one. Rebin, or restrict the template, if that is not what "
                "you meant.",
                stacklevel=2,
            )

        # every remaining bin is finite and non-zero, so the division below
        # cannot manufacture a NaN of its own
        self._null_safe = np.where(self._invalid, 1.0, null_binned)
        self._ones = np.ones_like(self._null_safe)

        # take care of correlated paramters
        self.new_pars = new_pars
        self.corr_pars, self.unco_pars = self._separate_pars(new_pars)
        self.corr_infos = self._corr_infos(self.corr_pars)

        # Weights already computed, keyed on the parameter point and ordered
        # least recently used first, so the bound can be enforced by dropping
        # from the front.
        self.cache = {}
        self.cache_size = cache_size

    def _tensorlib(self):
        """
        The active pyhf tensor library, checked against what HAMMER can do.

        `Modifier` allows a tracing backend where its quadrature can be traced.
        Nothing here can be: the templates come out of a compiled library called
        outside the tensor graph, so there is no derivative to propagate and no
        tracer to thread through it. Say so, rather than returning plain arrays
        that break further downstream.

        Returns:
            tensorlib: The active pyhf tensor library.

        Raises:
            ValueError: If the active backend is not numpy.
        """
        tensorlib, _ = get_backend()
        if tensorlib.name != "numpy":
            raise ValueError(
                f"the {tensorlib.name} backend cannot trace through HAMMER, "
                "which is a compiled library evaluated outside the tensor "
                "graph; the hammer modifier only runs on the numpy backend"
            )
        return tensorlib

    def get_weights(self, pars):
        """
        Compute the new weights and process them for sensibility.

        Args:
            pars (dict): pyhf parameters.

        Returns:
            array: Weights for the given parameters.
        """
        self._tensorlib()

        # compute original parameters from pyhf parameters
        rot_pars = self.rotate_pars(pars)
        alt_binned = np.asarray(self.alt_dist(**rot_pars), dtype=float)

        weights = alt_binned / self._null_safe
        weights = np.where(self._invalid, self._ones, weights)

        # The alternative template can be empty or not finite on its own. Both
        # NaN and infinity have to be caught: dividing a finite number by an
        # empty null bin gives infinity, not NaN, and an infinite weight
        # propagates straight into the yield.
        weights = np.where(np.isfinite(weights), weights, self._ones)
        if not self.allow_negative_weights:
            weights = np.where(weights < 0.0, self._ones, weights)
        # Presence, not truthiness. Testing the bound for truth silently
        # dropped a bound of zero, which is a bound like any other once
        # `allow_negative_weights` puts weights on both sides of it.
        if self.weight_bound is not None:
            weights = np.where(weights > self.weight_bound, self.weight_bound, weights)

        # flatten the weights in Fortran order, to match the sample's bin layout
        return weights.reshape(-1, order="F")

    def weight_func(self, pars):
        """
        Build function that applies weights to histogram.

        The templates are already in the analysis binning, so the weights are
        the modification: there is no mapping distribution to fold them through,
        as there is in `Modifier`.

        Args:
            pars (dict): pyhf parameters.

        Returns:
            callable: Function that returns histogram modifications.
        """
        cacheable = self.cache_size > 0
        if cacheable:
            key = tuple(i for i in pars.items())
            if key in self.cache:
                # reinsert, so the least recently used entry stays at the front
                self.cache[key] = self.cache.pop(key)
                return self.cache[key]

        results = self.get_weights(pars)

        def func():
            return results

        if cacheable:
            self.cache[key] = func
            if len(self.cache) > self.cache_size:
                del self.cache[next(iter(self.cache))]

        return func


def save_hammer(file, spec, cmods, data=None):
    """
    Save the custom model, mapping distribution (and data).

    Every setting that changes the weights is written out, so `load_hammer`
    rebuilds the same model. `cache_size` is not among them: it cannot change
    any result.

    Args:
        file (string): File name.
        spec (dict): Model specification.
        cmods (list): List of custom modifiers.
        data (array, optional): Data to be saved. Defaults to None.
    """
    d = {
        "spec": spec,
        "name": [cmod.name for cmod in cmods],
        "new_pars": [cmod.new_pars for cmod in cmods],
        "cutoff": [cmod.cutoff for cmod in cmods],
        "weight_bound": [cmod.weight_bound for cmod in cmods],
        "allow_negative_weights": [cmod.allow_negative_weights for cmod in cmods],
    }
    if data is not None:
        d["data"] = np.array(data).tolist()

    with open(file, "w") as f:
        json.dump(d, f, indent=4)


def load_hammer(
    file, alt_dist, null_dist, return_modifier=False, return_data=False, **kwargs
):
    """
    Load and build model from file

    Settings that `save_hammer` learned to write only later are optional: a file
    written before them omits the key and the modifier falls back to its own
    default, so older models load exactly as they did when they were saved.

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
    with open(file, "r") as f:
        d = json.load(f)

    new_pars = {}
    for pars in d["new_pars"]:
        new_pars.update(modifier._read_pars(pars))

    optional = {
        key: d.get(key, [None] * len(d["name"])) for key in ("allow_negative_weights",)
    }

    cmods = []
    for i, (name, cutoff, weight_bound) in enumerate(
        zip(d["name"], d["cutoff"], d["weight_bound"])
    ):
        saved = {k: v[i] for k, v in optional.items() if v[i] is not None}
        cmods.append(
            Modifier_Hammer(
                new_pars,
                alt_dist,
                null_dist,
                name=name,
                cutoff=cutoff,
                weight_bound=weight_bound,
                **saved,
            )
        )

    expanded_pyhf = {}
    for cmod in cmods:
        expanded_pyhf.update(cmod.expanded_pyhf)

    model = pyhf.Model(
        d["spec"], validate=False, batch_size=None, modifier_set=expanded_pyhf, **kwargs
    )

    if return_modifier and return_data:
        return model, cmods, d["data"]
    if return_modifier:
        return model, cmods
    if return_data:
        return model, d["data"]
    return model


# the hammer cacher class handles directly the hammer histogram
# it access it and it changes, if required, the FF and the WC d.o.f
# giving access to the histogram as it changes wrt them
class HammerCacher:
    """
    A single HAMMER histogram file, reweighted on demand.

    Reweighting a histogram is expensive, so the coefficients last asked for are
    remembered and the histogram is only rebuilt when they change.
    """

    def __init__(
        self,
        fileName,
        histoName,
        FFscheme,
        WilsonSet,
        FormFactors,
        WilsonCoefficients,
        scaleFactor,
        verbose=False,
    ):
        """
        Args:
            fileName (string): HAMMER histogram file.
            histoName (string): Name of the histogram inside it.
            FFscheme (dict): Form factor scheme, with `name`, `Process` and
                `SchemeVar` keys.
            WilsonSet (string): Name of the Wilson coefficient set.
            FormFactors (dict): Initial form factor eigenvector shifts.
            WilsonCoefficients (dict): Initial Wilson coefficients.
            scaleFactor (float): Yield the normalised histogram is scaled to.
            verbose (bool, optional): Report what is being read. Defaults to False.

        Raises:
            ImportError: If HAMMER is not installed.
        """
        Hammer, IOBuffer, RecordType = _hammerlib()

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
        if verbose:
            print(f"fileName = {fileName}")
            print(f"histoName = {histoName}")

        with open(fileName, "rb", buffering=0) as fin:
            if buf.load(fin) and self._ham.load_run_header(buf):
                self._ham.init_run()
                if buf.load(fin):
                    while buf.kind in (
                        RecordType.HISTOGRAM,
                        RecordType.HISTOGRAM_DEFINITION,
                    ):
                        if buf.kind == RecordType.HISTOGRAM_DEFINITION:
                            self._ham.load_histogram_definition(buf)
                        else:
                            self._ham.load_histogram(buf)
                        if not buf.load(fin):
                            break

        self._ham.set_ff_eigenvectors(
            self._FFScheme["Process"], self._FFScheme["SchemeVar"], self._FFs
        )
        self._ham.set_wilson_coefficients(self._WilsonSet, self._wcs)
        self._histo = self._ham.get_histogram(histoName, FFscheme["name"])

        # The histogram is flattened to one dimension. `_nobs` is its total bin
        # count; `_strides` is the step along each axis of the flat index, which
        # is what a consumer needs to project the histogram back onto a single
        # observable.
        shape = self._ham.get_histogram_shape(histoName)
        for ndim in shape:
            self._nobs *= ndim
        for dim in (shape[1:] + shape[:1])[:-1]:
            self._strides = [c * dim for c in self._strides]
            self._strides.append(1)

        self._normFactor = self.getHistoTotal()

    def checkWCCache(self, wcs):
        """
        Whether `wcs` is what the histogram already holds, updating the cache.

        Both at once, deliberately: the caller acts on the answer by rebuilding
        the histogram, and the remembered coefficients have to move with it.

        Args:
            wcs (dict): Wilson coefficients.

        Returns:
            bool: True if the coefficients were unchanged.
        """
        isCached = True
        for key in wcs:
            if key not in self._wcs.keys():
                self._wcs[key] = wcs[key]
                isCached = False
            elif not (self._wcs[key] - wcs[key]) == 0:
                self._wcs[key] = wcs[key]
                isCached = False
        return isCached

    def checkFFCache(self, FFs):
        """
        Whether `FFs` is what the histogram already holds, updating the cache.

        As `checkWCCache`, for the form factor eigenvector shifts.

        Args:
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            bool: True if the shifts were unchanged.
        """
        isCached = True
        for key in FFs:
            if key not in self._FFs.keys():
                self._FFs[key] = FFs[key]
                isCached = False
            elif not (self._FFs[key] - FFs[key]) == 0:
                self._FFs[key] = FFs[key]
                isCached = False
        return isCached

    def _refresh(self, wcs, FFs):
        """
        Bring the histogram up to date with the given coefficients.

        The two checks are what makes walking every bin affordable: only the
        first bin of a template pays for the reweighting, the rest find the
        histogram already built.

        Args:
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.
        """
        if not self.checkFFCache(FFs):
            self._ham.reset_ff_eigenvectors(
                self._FFScheme["Process"], self._FFScheme["SchemeVar"]
            )
            self._ham.set_ff_eigenvectors(
                self._FFScheme["Process"], self._FFScheme["SchemeVar"], FFs
            )
            self._histo = self._ham.get_histogram(
                self._histoName, self._FFScheme["name"]
            )
        if not self.checkWCCache(wcs):
            self._ham.reset_wilson_coefficients(self._WilsonSet)
            self._ham.set_wilson_coefficients(self._WilsonSet, wcs)
            self._histo = self._ham.get_histogram(
                self._histoName, self._FFScheme["name"]
            )

    def _histo_array(self):
        """
        The current histogram as a plain array, one entry per bin.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return np.fromiter(
            (self._histo[ni].sum_wi for ni in range(self._nobs)),
            dtype=float,
            count=self._nobs,
        )

    def getHistoTotal(self):
        """
        Total yield of the histogram as it currently stands.

        Returns:
            float: Sum over all bins.
        """
        return float(self._histo_array().sum())

    def getHistoTotalSM(self):
        """
        Total yield with only the standard model coefficient switched on.

        The coefficients in force are put back before returning. Leaving the
        standard model ones behind would make `_wcs` disagree with what HAMMER
        actually holds, and the cache check would then skip a reweighting it
        owed, silently handing back standard model yields under someone else's
        coefficients.

        Returns:
            float: Sum over all bins, in the standard model.
        """
        self._ham.reset_wilson_coefficients(self._WilsonSet)
        self._ham.set_wilson_coefficients(self._WilsonSet, _sm_only(self._wcs))
        self._histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        total = self.getHistoTotal()

        self._ham.reset_wilson_coefficients(self._WilsonSet)
        self._ham.set_wilson_coefficients(self._WilsonSet, self._wcs)
        self._histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        return total

    def getHistoArray(self, wcs, FFs):
        """
        Every bin at the given coefficients, unscaled.

        The cache check, and the reweighting it may trigger, happen once for the
        whole histogram rather than once per bin, which is what
        `getHistoElementByPosNoScale` amounts to when a caller walks all of them.

        Args:
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        self._refresh(wcs, FFs)
        return self._histo_array()

    def getHistoElementByPosNoScale(self, pos, wcs, FFs):
        """
        One bin at the given coefficients, unscaled.

        Args:
            pos (int): Flat bin index.
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            float: Content of the bin.
        """
        self._refresh(wcs, FFs)
        return self._histo[pos].sum_wi

    def getHistoElementByPosNoScaleSM(self, pos, wcs, FFs):
        """
        One bin with only the standard model coefficient switched on, unscaled.

        `wcs` is left alone; the coefficients are switched off in a copy.

        Args:
            pos (int): Flat bin index.
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            float: Content of the bin.
        """
        return self.getHistoElementByPosNoScale(pos, _sm_only(wcs), FFs)

    def getHistoArraySM(self, wcs, FFs):
        """
        Every bin with only the standard model coefficient switched on, unscaled.

        Args:
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return self.getHistoArray(_sm_only(wcs), FFs)


# Multi hammer cacher allows you to store multiple histograms in multiple files
# and treat them like a single one for when you parallelize the hammer reweighting process
# that can be very time consuming
class MultiHammerCacher:
    """
    Several `HammerCacher` histograms added together and treated as one.

    Reweighting is slow enough to be worth splitting over several files; this
    puts them back together, normalised to the common scale factor.
    """

    def __init__(self, cacherList):
        """
        Args:
            cacherList (list): Cachers over the same histogram definition.
        """
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
            self._normFactor += cacher.getHistoTotal()

    def getHistoElementByPos(self, pos, wcs, FFs):
        """
        One bin, summed over the files and scaled.

        Args:
            pos (int): Flat bin index.
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            float: Content of the bin.
        """
        res = 0
        for cacher in self._cacherList:
            res += cacher.getHistoElementByPosNoScale(pos, wcs, FFs)
        self._wcs = wcs
        self._FFs = FFs
        return res * self._scaleFactor / self._normFactor

    def getHistoElementByPosSM(self, pos, wcs, FFs):
        """
        One bin in the standard model, summed over the files and scaled.

        `wcs` is left alone; the coefficients are switched off in a copy.

        Args:
            pos (int): Flat bin index.
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            float: Content of the bin.
        """
        return self.getHistoElementByPos(pos, _sm_only(wcs), FFs)

    def getHistoArray(self, wcs, FFs):
        """
        Every bin, summed over the files and scaled.

        Args:
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        res = np.zeros(self._nobs)
        for cacher in self._cacherList:
            res = res + cacher.getHistoArray(wcs, FFs)
        self._wcs = wcs
        self._FFs = FFs
        return res * self._scaleFactor / self._normFactor

    def getHistoArraySM(self, wcs, FFs):
        """
        Every bin in the standard model, summed over the files and scaled.

        Args:
            wcs (dict): Wilson coefficients.
            FFs (dict): Form factor eigenvector shifts.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return self.getHistoArray(_sm_only(wcs), FFs)


# the background cacher access not hammer reweighted histograms and gives us in a format
# similar to the HammerCacher (easier to handle them together later)
class BackgroundCacher:
    """
    A histogram that is not HAMMER reweighted, in the shape of a `HammerCacher`.

    Backgrounds do not depend on the coefficients, so the `wcs` and `FFs`
    arguments are accepted and ignored. Having the same interface is what lets a
    wrapper treat the two alike.
    """

    def __init__(self, fileName, histoName, strides):
        """
        Args:
            fileName (string): Text file holding one bin content per line.
            histoName (string): Name of the histogram, kept for reference.
            strides (list): Step along each axis of the flat bin index.

        Raises:
            ValueError: If the file holds no bins.
        """
        self._fileName = fileName
        self._histoName = histoName
        self._strides = strides
        # Let the read fail on its own. This used to print the exception and
        # return, leaving an object without `_nobs` or `_normFactor`, so the
        # real fault surfaced much later as an unrelated AttributeError.
        self._histo = np.loadtxt(self._fileName)
        if len(self._histo) == 0:
            raise ValueError(f"the background histogram in {self._fileName} is empty")
        self._nobs = len(self._histo)
        self._normFactor = self._histo.sum()

    def getHistoElementByPos(self, pos, wcs, FFs):
        """
        One normalised bin.

        Args:
            pos (int): Flat bin index.
            wcs (dict): Ignored.
            FFs (dict): Ignored.

        Returns:
            float: Content of the bin.
        """
        return self._histo[pos] / self._normFactor

    def getHistoArray(self, wcs, FFs):
        """
        Every normalised bin.

        Args:
            wcs (dict): Ignored.
            FFs (dict): Ignored.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return self._histo / self._normFactor


class _HammerNuisWrapperBase:
    """
    Everything `HammerNuisWrapper` and `HammerNuisWrapperSM` share.

    The two differ only in whether the Wilson coefficients reach the cacher, so
    the state and the setters live here and each subclass supplies its own
    `evaluate`. It is a private base rather than one wrapper inheriting from the
    other, so that neither is an instance of the other, as was the case when the
    two were written out in full.
    """

    def __init__(self, hac, **kwargs):
        """
        Args:
            hac: Cacher to wrap.
            kwargs: Nuisance parameters and their initial values.
        """
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

    def set_wcs(self, wcs):
        """
        Set the Wilson coefficients from their real and imaginary parts.

        pyhf only ever varies real parameters, so a complex coefficient arrives
        as a `Re_` and an `Im_` key, which are paired back up here by name.

        Args:
            wcs (dict): Coefficients, as `Re_`/`Im_` pairs or plain reals.
        """
        self._wcs = {}
        for key in wcs.keys():
            if key.startswith("Re_"):
                base_name = key[3:]  # Remove "Re_" prefix
                im_key = "Im_" + base_name
                if im_key in wcs:  # Ensure both Re_ and Im_ exist
                    self._wcs[base_name] = complex(wcs[key], wcs[im_key])
            elif not key.startswith("Im_"):  # Avoid adding "Im_" keys separately
                self._wcs[key] = wcs[key]

    def set_FFs(self, FFs):
        """
        Set the form factor eigenvector shifts, ignoring any unknown name.

        Args:
            FFs (dict): Eigenvector shifts.
        """
        FFs_temp = {}
        for key, value in FFs.items():
            if key in self._FFs.keys():
                FFs_temp[key] = float(value)
        self._FFs = FFs_temp

    def set_params(self, params):
        """
        Set the nuisance parameters, ignoring any unknown name.

        Args:
            params (dict): Nuisance parameters.
        """
        params_temp = {}
        for key, value in params.items():
            if key in self._params.keys():
                params_temp[key] = value
        self._params = params_temp

    def set_nbin(self, nbin):
        """
        Select the bin `evaluate` reports.

        Args:
            nbin (int): Flat bin index.
        """
        self._nbin = nbin

    def _scale(self, val):
        """
        Apply the nuisance parameters.

        Args:
            val: Bin content, or an array of them.

        Returns:
            The content scaled by every nuisance parameter.
        """
        for value in self._params.values():
            val = val * value
        return val


# here we define the multiplicative nuisance parameters to apply to the hammer
# reweighted histogram
# the wrapper can change the d.o.f of the contribution and returns the content of
# a given bin wrt to the current values of the d.o.f. with an evaluate function
class HammerNuisWrapper(_HammerNuisWrapperBase):
    """
    A cacher plus the multiplicative nuisance parameters applied on top of it.

    Holds the degrees of freedom -- coefficients, form factor shifts and
    nuisances -- and reports what the histogram looks like at their current
    values.
    """

    def evaluate(self):
        """
        The selected bin at the current degrees of freedom.

        Returns:
            float: Content of the bin.
        """
        return self._scale(
            self._hac.getHistoElementByPos(self._nbin, self._wcs, self._FFs)
        )

    def evaluate_all(self):
        """
        Every bin at the current degrees of freedom.

        The same numbers `evaluate` gives bin by bin, in the same order, but the
        cacher is asked once instead of once per bin.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return self._scale(self._hac.getHistoArray(self._wcs, self._FFs))


# this is the same as HammerNuisWrapper but the evaluate method is
# ignoring the WCs
# if you for example want to inject new physics B2DTauNu and not in B2DMuNu
# you'll build a ordinary HammerNuisWrapper for B2DTauNu
# and a SM one for B2DMuNu
class HammerNuisWrapperSM(_HammerNuisWrapperBase):
    """
    As `HammerNuisWrapper`, but always evaluated in the standard model.

    For a mode that should not receive the new physics being injected
    elsewhere: a B2DTauNu signal gets the ordinary wrapper, a B2DMuNu
    normalisation gets this one.
    """

    def evaluate(self):
        """
        The selected bin in the standard model.

        Returns:
            float: Content of the bin.
        """
        return self._scale(
            self._hac.getHistoElementByPosSM(self._nbin, self._wcs, self._FFs)
        )

    def evaluate_all(self):
        """
        Every bin in the standard model.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return self._scale(self._hac.getHistoArraySM(self._wcs, self._FFs))


# this attaches Nuisance parameters to the BackgroundCacher
# notice that one of the Nuisance parameters here should always be the yield
class BackgroundNuisWrapper:
    """
    A `BackgroundCacher` plus its multiplicative nuisance parameters.

    One of those parameters should always be the yield, since the cacher hands
    back a normalised histogram.
    """

    def __init__(self, bkg, **kwargs):
        """
        Args:
            bkg: Background cacher to wrap.
            kwargs: Nuisance parameters and their initial values.
        """
        self._bkg = bkg
        self._nobs = bkg._nobs
        self._params = {}
        self._wcs = {}
        self._FFs = {}
        for key, value in kwargs.items():
            self._params[key] = value
        self._nbin = 0
        self._strides = bkg._strides
        self._dim = len(bkg._strides)

    def set_nbin(self, nbin):
        """
        Select the bin `evaluate` reports.

        Args:
            nbin (int): Flat bin index.
        """
        self._nbin = nbin

    def set_wcs(self, wcs):
        """
        Ignore the Wilson coefficients: a background does not depend on them.

        Args:
            wcs (dict): Ignored.
        """
        self._wcs = {}

    def set_FFs(self, FFs):
        """
        Ignore the form factors: a background does not depend on them.

        Args:
            FFs (dict): Ignored.
        """
        self._FFs = {}

    def set_params(self, params):
        """
        Set the nuisance parameters, ignoring any unknown name.

        Args:
            params (dict): Nuisance parameters.
        """
        params_temp = {}
        for key, value in params.items():
            if key in self._params.keys():
                params_temp[key] = value
        self._params = params_temp

    def _scale(self, val):
        """
        Apply the nuisance parameters.

        Args:
            val: Bin content, or an array of them.

        Returns:
            The content scaled by every nuisance parameter.
        """
        for value in self._params.values():
            val = val * value
        return val

    def evaluate(self):
        """
        The selected bin at the current nuisance parameters.

        Returns:
            float: Content of the bin.
        """
        return self._scale(
            self._bkg.getHistoElementByPos(self._nbin, self._wcs, self._FFs)
        )

    def evaluate_all(self):
        """
        Every bin at the current nuisance parameters.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return self._scale(self._bkg.getHistoArray(self._wcs, self._FFs))


# the template class takes the wrapper and allows to generate templates, and toys
# wrt any set of d.o.f we want
class template:
    """
    A wrapper turned into binned yields, for any set of degrees of freedom.
    """

    def __init__(self, name, wrap):
        """
        Args:
            name (string): Name of the mode.
            wrap: Wrapper to generate from.
        """
        self._name = name
        self._wrap = wrap
        self._nobs = wrap._nobs
        self._nwcs = len(self._wrap._wcs)
        self._nFFs = len(self._wrap._FFs)
        self._nparams = len(self._wrap._params)
        self._strides = wrap._strides

        # Which keyword belongs to which group, worked out once from the
        # wrapper. A complex coefficient reaches us split into `Re_` and `Im_`
        # halves, so both spellings count as the coefficient they belong to.
        self._wc_names = set()
        for key in self._wrap._wcs:
            self._wc_names.update({key, "Re_" + key, "Im_" + key})
        self._FF_names = set(self._wrap._FFs)

    def _split_pars(self, kwargs):
        """
        Route each keyword to the coefficients, the form factors or the nuisances.

        By name, not by position. This used to slice the keywords at offsets
        derived from the number of coefficients, so it depended on the order the
        caller happened to pass them in, and on the standard model being the one
        real coefficient among complex ones; reordering a call silently
        reinterpreted the values as something else.

        Anything the wrapper does not know as a coefficient or a form factor is
        taken to be a nuisance parameter, and the wrapper drops it if it does
        not know it as one of those either.

        Args:
            kwargs (dict): Degrees of freedom, by name.

        Returns:
            tuple: Coefficients, form factors and nuisance parameters.
        """
        wcs, FFs, params = {}, {}, {}
        for key, value in kwargs.items():
            if key in self._wc_names:
                wcs[key] = value
            elif key in self._FF_names:
                FFs[key] = value
            else:
                params[key] = value
        return wcs, FFs, params

    def _evaluate(self, kwargs):
        """
        Bin contents at the given degrees of freedom.

        Args:
            kwargs (dict): Degrees of freedom, by name.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        wcs, FFs, params = self._split_pars(kwargs)
        self._wrap.set_wcs(wcs)
        self._wrap.set_FFs(FFs)
        self._wrap.set_params(params)

        evaluate_all = getattr(self._wrap, "evaluate_all", None)
        if evaluate_all is not None:
            return np.asarray(evaluate_all(), dtype=float)

        # a wrapper from outside this module may only offer the per-bin call
        bin_contents = np.zeros(self._nobs)
        for i in range(self._nobs):
            self._wrap.set_nbin(i)
            bin_contents[i] = self._wrap.evaluate()
        return bin_contents

    def generate_template(self, **kwargs):
        """
        Expected yields at the given degrees of freedom.

        Args:
            kwargs: Coefficients, form factors and nuisance parameters, by name.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return self._evaluate(kwargs)

    def generate_toy(self, **kwargs):
        """
        A Poisson draw around the expected yields.

        Args:
            kwargs: Coefficients, form factors and nuisance parameters, by name.

        Returns:
            array: Bin contents, of length `_nobs`.
        """
        return np.random.poisson(self._evaluate(kwargs)).astype(float)


# the fitter contains a template list and data (toys in the examples)
# it contains the definition of a nul_pdf and an alternative_pdf to be injected in the definition of the modifier
# a small plotting interface is implemented to retireve the projected histograms (from the strides) and overlay data
class fitter:
    """
    The templates of every mode, and the data they are to be fitted to.
    """

    def __init__(self, template_list):
        """
        Args:
            template_list (list): Templates, in the order the config declared them.
        """
        self._template_list = template_list
        self._data = np.array([])

    def get_template(self, index):
        """
        Args:
            index (int): Position of the mode in the config.

        Returns:
            template: The template for that mode.
        """
        return self._template_list[index]

    def upload_data(self, data):
        """
        Args:
            data (array): Observed yields.
        """
        self._data = data


# The reader class aim is to make everything above not necessary to be fully undestood
# A config file is provided and the reader produces itself the necessary objects:
# Cachers -> Wrappers -> Templates -> Fitter (returned)
# giving access to a fitter with a toy stored inside as data (temporary)
class Reader:
    """
    Builds the whole chain -- cachers, wrappers, templates, fitter -- from a config file.
    """

    def __init__(self, filename):
        """
        Args:
            filename (string): JSON config, one entry per mode.
        """
        self.name = filename
        with open(filename, "r") as f:
            self.config = json.load(f)

    def createFitter(self, verbose=False):
        """
        Build the fitter the config describes.

        Args:
            verbose (bool, optional): Report what is being read. Defaults to False.

        Returns:
            fitter: One template per mode in the config.
        """
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
                _wilsoncoefficients[key] = complex(value[0], value[1])

            if is_hammer_weighted:
                for fileName in fileNames:
                    if verbose:
                        print(f"Reading {fileName}")
                    # a cacher of its own for each file, since each one keeps
                    # its coefficients as mutable state
                    hac_list.append(
                        HammerCacher(
                            fileName,
                            histoname,
                            ffscheme,
                            wcscheme,
                            deepcopy(formfactors),
                            deepcopy(_wilsoncoefficients),
                            scalefactor,
                        )
                    )
                cacher = MultiHammerCacher(hac_list)
                wrapper_type = HammerNuisWrapper if injectNP else HammerNuisWrapperSM
                wrapper = wrapper_type(cacher, **nuisance)
            else:
                for fileName in fileNames:
                    if verbose:
                        print(f"Reading {fileName}")
                    hac_list.append(BackgroundCacher(fileName, histoname, strides))
                wrapper = BackgroundNuisWrapper(hac_list[0], **nuisance)

            template_list.append(template(mode, wrapper))

        return fitter(template_list)
