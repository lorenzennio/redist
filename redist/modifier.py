import re
from copy import deepcopy
import itertools
from collections import defaultdict
from collections.abc import Iterable
import numpy as np
import scipy as sp
import json
import pyhf
from pyhf import get_backend
from redist import custom_modifier


class Modifier:
    """
    Modifier implementation to reweight historgram according to the ratio of
    a null and an alternative distribution.
    """

    def __init__(
        self,
        new_pars,
        alt_dist,
        null_dist,
        map,
        bins,
        name=None,
        cutoff=None,
        weight_bound=None,
        allow_negative_weights=False,
        quad="auto",
        quad_order=16,
    ):
        """
        Args:
            new_pars (dict): New parameters to parametrize the model.
            alt_dist (callable): Alternative distribution to be tested.
            null_dist (callable): Null distribution of the nominal model.
            map (array): Joint number density matrix, binned in the analysis bins times the kinematic bins.
            bins (array): kinematic binning
            name (string, optional): Name of the custom modifier. Defaults to None.
            cutoff (tuple, optional): Kinematic cutoff values to limit the integration boundaries to a given range. Defaults to None.
            weight_bound (float, optional): Upper bound on the weight. Defaults to None.
            allow_negative_weights (bool, optional): Allow negative weights. Defaults to False.
            quad (string, optional): Quadrature rule for the bin integrals.
                ``"nquad"`` uses adaptive `scipy` quadrature and only works on
                the NumPy backend. ``"gauss"`` uses fixed-order Gauss-Legendre
                quadrature, which can be traced and differentiated, and requires
                the distributions to accept broadcast arrays. ``"auto"`` picks
                `nquad` on the NumPy backend and `gauss` on any other, based on
                the backend active at construction. Defaults to ``"auto"``.
            quad_order (int, optional): Gauss-Legendre nodes per bin and
                dimension. Ignored by `nquad`. Defaults to 16.

        Raises:
            ValueError: If the null distribution integrates to zero, or to
                something not finite, in any bin inside the cutoff. The
                reweighting ratio is undefined there, so no physical yield can
                be built from it.
        """
        # store name and cutoff
        self.name = name if name else "custom"
        self.cutoff = cutoff
        self.weight_bound = weight_bound
        self.allow_negative_weights = allow_negative_weights

        # store null and alternative distributions
        self.null_dist = null_dist
        self.alt_dist = alt_dist

        # store mapping distribution and binning
        shape = np.shape(map)
        self.map = np.reshape(map, (shape[0], np.prod(shape[1:])))
        self.bins = bins

        self.nominal = np.sum(self.map, axis=1)

        # Resolve the quadrature rule once, so the null and the alternative are
        # always integrated the same way. "auto" reads the backend active now,
        # so set the backend before building the modifier.
        if quad == "auto":
            quad = "nquad" if get_backend()[0].name == "numpy" else "gauss"
        if quad not in ("nquad", "gauss"):
            raise ValueError(
                f"unknown quadrature rule {quad!r}, expected auto, nquad or gauss"
            )
        self.quad = quad
        self.quad_order = quad_order

        # `nquad` marks bins outside the cutoff with NaN, and the weights below
        # turn those into ones. `gauss` can do the same, but the NaN would make
        # the gradient NaN even though the value is discarded, so there the
        # cutoff is applied through `_invalid` instead and never reaches the
        # integrator.
        self._quad_cutoff = self.cutoff if self.quad == "nquad" else None

        # compute the bin-integrated null distribution (this is fixed)
        self.null_binned = bintegrate(
            null_dist,
            bins,
            cutoff=self._quad_cutoff,
            quad=self.quad,
            order=self.quad_order,
        )

        # Bins the cutoff excludes carry no information and are dropped. This
        # follows from the binning alone, so it is settled here rather than
        # rediscovered from NaNs on every call, which also keeps NaN out of the
        # traced path where it would poison gradients.
        null_binned = np.asarray(self.null_binned, dtype=float)
        self._invalid = ~_inside_cutoff(bins, self.cutoff)

        # A bin the null distribution does not populate cannot be reweighted:
        # the ratio has no finite value, and any yield built from it would be
        # unphysical rather than merely imprecise. Refuse to build the modifier
        # instead of substituting something that looks like a result.
        degenerate = ~self._invalid & ~(np.isfinite(null_binned) & (null_binned != 0.0))
        if degenerate.any():
            raise ValueError(
                "the null distribution integrates to zero or is not finite in "
                f"{int(degenerate.sum())} of {int((~self._invalid).sum())} bins "
                "inside the cutoff, so the reweighting ratio is undefined there "
                "and the yields would not be physical. Affected bins: "
                f"{', '.join(_describe_bins(bins, degenerate))}. Restrict the "
                "binning or the cutoff to where the null distribution has "
                "support."
            )

        # every remaining bin is finite and non-zero, so the division below
        # cannot manufacture a NaN of its own
        self._null_safe = np.where(self._invalid, 1.0, null_binned)
        self._ones = np.ones_like(self._null_safe)

        # take care of correlated paramters
        self.new_pars = new_pars
        self.corr_pars, self.unco_pars = self._separate_pars(new_pars)
        self.corr_infos = self._corr_infos(self.corr_pars)

        # cache previously called function values
        self.cache = {}

    @property
    def expanded_pyhf(self):
        """
        Build expanded pyhf modifier set
        """
        return custom_modifier.add(
            self.name,
            list(self.unco_pars.keys()),
            self.unco_pars,
            namespace={self.name + "_weight_fn": self.weight_func},
        )

    def _separate_pars(self, new_pars):
        """
        Separate parameters into correlated and uncorrelated ones.

        Args:
            new_pars (dict): New parameters to parametrize the model.

        Returns:
            dict, dict: Correlated and uncorrelated parameters.
        """
        corr_pars = {}
        unco_pars = {}
        for k, v in new_pars.items():
            if "cov" in v.keys():
                corr_pars[k] = v
                # for each correlated parameter, add one pyhf parameter
                for n, _ in enumerate(v["inits"]):
                    name = k + f"_decorrelated[{n}]"
                    unco_pars[name] = {
                        "inits": (0.0,),
                        "bounds": ((-5.0, 5.0),),
                        "paramset_type": v["paramset_type"],
                    }
            else:
                unco_pars[k] = v

        return corr_pars, unco_pars

    def _corr_infos(self, corr_pars):
        """
        Compute and store svd rotation matrix for correlated parameters.

        Args:
            corr_pars (dict): Subset of `new_pars` containing correlated parameters.

        Returns:
            dict: Dictionary containing the mean and rotation matrix for each correlated parameter.
        """
        corr_infos = {}
        if corr_pars:
            for k, v in corr_pars.items():
                corr_infos[k] = {"mean": v["inits"], "uvec": _svd(v["cov"])}

        return corr_infos

    def rotate_pars(self, pars):
        """
        Map from svd parameters to true parameters.

        Args:
            pars (dict): pyhf parameters.

        Returns:
            dict: Rotated parameters.
        """
        rot_pars = {}
        for k, v in pars.items():
            rot_pars[re.sub("_decorrelated", "", k)] = v
        pyhf_shifts = defaultdict(list)

        for corr_k in self.corr_infos:
            for par_k, par_v in pars.items():
                if corr_k == re.sub(r"_decorrelated[\(\[].*?[\)\]]", "", par_k):
                    pyhf_shifts[corr_k].append(par_v)

        tensorlib = self._tensorlib()
        for corr_k, pyhf_shift_list in pyhf_shifts.items():
            corr_v = self.corr_infos[corr_k]
            pyhf_shifts_arr = tensorlib.stack(pyhf_shift_list)
            pars_shifts = tensorlib.astensor(corr_v["uvec"]) @ pyhf_shifts_arr
            pars_new = tensorlib.astensor(corr_v["mean"]) + pars_shifts
            for ind, par in enumerate(pars_new):
                rot_pars[corr_k + f"[{ind}]"] = par

        return rot_pars

    def _tensorlib(self):
        """
        The active pyhf tensor library, checked against the quadrature rule.

        The fixed arrays are deliberately kept as plain NumPy and converted at
        the point of use rather than cached per backend. A cached array that was
        first built inside a `jax.jit` trace is a tracer, and reusing it on the
        next call leaks it out of its trace.

        Returns:
            tensorlib: The active pyhf tensor library.
        """
        tensorlib, _ = get_backend()
        if tensorlib.name != "numpy" and self.quad == "nquad":
            raise ValueError(
                f"the {tensorlib.name} backend cannot trace through adaptive "
                "quadrature; build the modifier with quad='gauss', or set the "
                "backend before building it so quad='auto' can pick it up"
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
        tensorlib = self._tensorlib()

        # compute original parameters from pyhf parameters
        rot_pars = self.rotate_pars(pars)

        alt_binned = bintegrate(
            self.alt_dist,
            self.bins,
            tuple(rot_pars.values()),
            cutoff=self._quad_cutoff,
            quad=self.quad,
            order=self.quad_order,
        )

        weights = tensorlib.divide(alt_binned, self._null_safe)
        weights = tensorlib.where(self._invalid, self._ones, weights)

        # an alternative distribution can still return NaN on its own; the
        # comparison is the backend-agnostic spelling of isnan
        weights = tensorlib.where(weights != weights, self._ones, weights)
        if not self.allow_negative_weights:
            weights = tensorlib.where(weights < 0.0, self._ones, weights)
        if self.weight_bound:
            weights = tensorlib.where(
                weights > self.weight_bound,
                self._ones * self.weight_bound,
                weights,
            )

        # flatten the weights in Fortran order, to match the map's column
        # layout; transposing then ravelling is the backend-agnostic spelling
        return tensorlib.ravel(tensorlib.transpose(weights))

    def weight_func(self, pars):
        """
        Build function that applies weights to histogram.

        Args:
            pars (dict): pyhf parameters.

        Returns:
            callable: Function that returns histogram modifications.
        """
        tensorlib = self._tensorlib()

        # Only NumPy hands over concrete, hashable parameter values, and it is
        # also the backend that pays for adaptive quadrature. A tracing backend
        # would key the cache on a tracer, so skip it there and let jit do the
        # equivalent job.
        cacheable = tensorlib.name == "numpy"
        if cacheable:
            key = tuple(i for i in pars.items())
            if key in self.cache:
                return self.cache[key]

        weights = self.get_weights(pars)
        results = tensorlib.astensor(self.map) @ weights
        results = results / self.nominal

        def func():
            return results

        if cacheable:
            self.cache[key] = func

        return func


def bintegrate(func, bins, args=(), cutoff=None, quad="nquad", order=16):
    """
    Integrate function in given bins.

    Args:
        func (callable): Function to be integrated.
        bins (array): Binning of the integration.
        args (tuple, optional): Additional arguments for the function. Defaults to ().
        cutoff (tuple, optional): Cutoff values for the integration. Defaults to None.
        quad (str, optional): Quadrature rule, ``"nquad"`` or ``"gauss"``.
            Defaults to ``"nquad"``.
        order (int, optional): Number of Gauss-Legendre nodes per bin and
            dimension, ignored by ``"nquad"``. Defaults to 16.

    Returns:
        array: Bin-integrated function values.
    """
    if quad == "gauss":
        return _bintegrate_gauss(func, bins, args=args, cutoff=cutoff, order=order)
    if quad != "nquad":
        raise ValueError(f"unknown quadrature rule {quad!r}, expected nquad or gauss")

    cutoff = cutoff if cutoff else tuple((-np.inf, np.inf) for _ in bins)
    ranges = [list(zip(b[:-1], b[1:])) for b in bins]
    results = []
    for limits in itertools.product(*ranges):
        # enforce cutoff
        if any(
            limit[0] < cut[0] or limit[1] > cut[1] for limit, cut in zip(limits, cutoff)
        ):
            results.append(np.nan)
        else:
            results.append(sp.integrate.nquad(func, limits, args=args)[0])
    return np.reshape(results, tuple(len(b) - 1 for b in bins)).T


def _bintegrate_gauss(func, bins, args=(), cutoff=None, order=16):
    """
    Integrate function in given bins by tensor-product Gauss-Legendre quadrature.

    Unlike `scipy.integrate.nquad`, this evaluates `func` at a fixed set of
    points, so it can be traced and differentiated. It is exact for polynomials
    up to degree ``2 * order - 1`` per dimension.

    `func` is called once, with one broadcast array per kinematic dimension, all
    of the same shape, and must return an array of that shape. That is the plain
    elementwise convention; it does not accept the scalar-at-a-time signature
    `nquad` uses.

    Every bin is integrated, including those outside the cutoff, which are then
    marked with NaN to match `nquad`. Callers that need to differentiate through
    the result should leave `cutoff` unset and exclude those bins themselves,
    since a NaN reaching the graph makes the gradient NaN even where the value
    is later discarded. `Modifier` does exactly that.

    Args:
        func (callable): Function to be integrated.
        bins (array): Binning of the integration.
        args (tuple, optional): Additional arguments for the function. Defaults to ().
        cutoff (tuple, optional): Cutoff values for the integration. Defaults to None.
        order (int, optional): Nodes per bin and dimension. Defaults to 16.

    Returns:
        array: Bin-integrated function values.
    """
    tensorlib, _ = get_backend()

    # Nodes and weights are fixed, and mapping them onto the bins depends only
    # on the binning, so all of this is plain NumPy regardless of the backend.
    nodes, node_weights = np.polynomial.legendre.leggauss(order)
    axis_nodes = []
    axis_weights = []
    for b in bins:
        edges = np.asarray(b, dtype=float)
        low = edges[:-1, None]
        high = edges[1:, None]
        half = 0.5 * (high - low)
        axis_nodes.append((0.5 * (low + high) + half * nodes).ravel())
        axis_weights.append((half * node_weights).ravel())

    grid = np.meshgrid(*axis_nodes, indexing="ij")
    weight_grid = np.meshgrid(*axis_weights, indexing="ij")
    quad_weights = weight_grid[0]
    for w in weight_grid[1:]:
        quad_weights = quad_weights * w

    integrand = func(*grid, *args) * quad_weights

    # Split every axis back into (bin, node) and sum the nodes away. Walking the
    # dimensions backwards keeps the axis numbers of the remaining ones valid.
    split_shape = []
    for b in bins:
        split_shape += [len(b) - 1, order]
    integrand = tensorlib.reshape(integrand, tuple(split_shape))
    for dim in reversed(range(len(bins))):
        integrand = tensorlib.sum(integrand, axis=2 * dim + 1)

    # match the axis order `bintegrate` returns
    result = tensorlib.transpose(integrand)

    if cutoff is not None:
        result = tensorlib.where(_inside_cutoff(bins, cutoff), result, np.nan)
    return result


def _describe_bins(bins, mask, limit=3):
    """
    Edges of the flagged bins, for error messages.

    `bintegrate` returns its result transposed, so the last axis is the first
    kinematic dimension.

    Args:
        bins (array): Binning of the integration.
        mask (array): Boolean array in `bintegrate`'s layout.
        limit (int, optional): Most bins to describe. Defaults to 3.

    Returns:
        list: Human-readable bin ranges.
    """
    flagged = np.argwhere(np.atleast_1d(mask))
    described = []
    for index in flagged[:limit]:
        edges = []
        for dim, edge_array in enumerate(bins):
            edge_array = np.asarray(edge_array, dtype=float)
            position = index[-1 - dim]
            edges.append(f"[{edge_array[position]:g}, {edge_array[position + 1]:g}]")
        described.append(" x ".join(edges))
    if len(flagged) > limit:
        described.append(f"and {len(flagged) - limit} more")
    return described


def _inside_cutoff(bins, cutoff):
    """
    Bins lying fully inside the cutoff, in the layout `bintegrate` returns.

    `bintegrate` marks excluded bins with NaN, which is enough for NumPy but
    would poison gradients under a tracing backend. Deriving the same
    information from the binning alone keeps NaN out of the graph.

    Args:
        bins (array): Binning of the integration.
        cutoff (tuple): Cutoff values, or None for no cutoff.

    Returns:
        array: Boolean array, True where the bin is inside the cutoff.
    """
    cutoff = cutoff if cutoff else tuple((-np.inf, np.inf) for _ in bins)
    ranges = [list(zip(b[:-1], b[1:])) for b in bins]
    inside = [
        all(
            limit[0] >= cut[0] and limit[1] <= cut[1]
            for limit, cut in zip(limits, cutoff)
        )
        for limits in itertools.product(*ranges)
    ]
    return np.reshape(inside, tuple(len(b) - 1 for b in bins)).T


def _svd(cov, return_rot=False):
    """Singular value decomposition, moving to a space where the covariance matrix is diagonal
    https://www.cs.cmu.edu/~elaw/papers/pca.pdf

    Args:
        cov (array): Covariance matrix

    Returns:
        array: matrix of column wise error vectors (eigenvectors * sqrt(eigenvalues); sqrt(eigenvalues) = std)
    """
    if len(cov) == 1:
        rot = np.array([[1.0]])
        uvec = np.sqrt(cov)
    else:
        svd = np.linalg.svd(cov)
        rot = svd[0]
        uvec = svd[0] @ np.sqrt(np.diag(svd[1]))

    if return_rot:
        return uvec, rot
    return uvec


def par_dict(model, pars):
    """
    Build parameter dictionary for pyhf model.

    Args:
        model (pyhf.Model): pyhf model.
        pars (dict): Parameters.

    Returns:
        dict: Dictionary of parameters by names.
    """
    try:
        par_list = pars.tolist()
    except AttributeError:
        par_list = pars

    return {
        k: par_list[v["slice"]][0]
        if len(par_list[v["slice"]]) == 1
        else par_list[v["slice"]]
        for k, v in model.config.par_map.items()
    }


def add_to_model(
    model, channels, samples, modifier_set, modifier_specs, **model_kwargs
):
    """
    Add a custom modifier to a pyhf model.

    Args:
        model (pyhf.Model): pyhf model.
        channels (list): List of channel names to add the modifier to.
        samples (list): List of sample names to add the modifier to.
        modifier_set (pyhf.modifier.ModifierSet): Pyhf modifier set.
        modifier_specs (dict): Modifier specifications.
        model_kwargs (dict): Additional model arguments.

    Returns:
        pyhf.Model: Model with the custom modifier added.
    """
    spec = model.spec

    for c, chan in enumerate(spec["channels"]):
        if chan["name"] in channels:
            for s, samp in enumerate(chan["samples"]):
                if samp["name"] in samples:
                    spec["channels"][c]["samples"][s]["modifiers"].append(
                        modifier_specs
                    )

    model = pyhf.Model(
        spec, validate=False, batch_size=None, modifier_set=modifier_set, **model_kwargs
    )

    return model


def save(file, spec, cmods, data=None):
    """
    Save the custom model, mapping distribution (and data).

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
        "map": [cmod.map.tolist() for cmod in cmods],
        # one binning array per kinematic dimension; converting each dimension
        # separately covers a list of arrays, which is what `Modifier` is
        # normally handed, as well as a list of lists or a single 2d array
        "bins": [[np.asarray(b).tolist() for b in cmod.bins] for cmod in cmods],
        "cutoff": [cmod.cutoff for cmod in cmods],
        "weight_bound": [cmod.weight_bound for cmod in cmods],
    }
    if data is not None:
        d["data"] = np.array(data).tolist()

    with open(file, "w") as f:
        json.dump(d, f, indent=4)


def load(file, alt_dist, null_dist, return_modifier=False, return_data=False, **kwargs):
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
    with open(file, "r") as f:
        d = json.load(f)

    new_pars = {}
    for pars in d["new_pars"]:
        new_pars.update(_read_pars(pars))
    cmods = []
    for name, map, bins, cutoff, weight_bound in zip(
        d["name"], d["map"], d["bins"], d["cutoff"], d["weight_bound"]
    ):
        cmods.append(
            Modifier(
                new_pars,
                alt_dist,
                null_dist,
                map,
                bins,
                name=name,
                cutoff=cutoff,
                weight_bound=weight_bound,
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


class _UnvalidatedWorkspace(pyhf.Workspace):
    """
    Workspace that skips schema validation.

    Custom modifier types are not part of pyhf's workspace schema, so a
    workspace holding one cannot be validated. Both `Workspace.build` and
    `Workspace.combine` construct their result through `cls`, so defaulting
    validation off here carries through both of them.
    """

    def __init__(self, spec, validate=False, **config_kwargs):
        super().__init__(spec, validate=validate, **config_kwargs)


def combine(files, alt_dists, null_dists, return_data=False, **kwargs):
    """
    Combine multiple models into one.

    Args:
        files (list): List of file names containing pyhf models to be combined.
        alt_dists (list): List of alternative distributions.
        null_dists (list): List of null distributions.
        return_data (bool, optional): Return data. Defaults to False.
        kwargs: Additional arguments for the pyhf model.

    Returns:
        pyhf.Model, array: Model, data.
    """
    models = []
    cmods = []
    datas = []
    for f, a, n in zip(files, alt_dists, null_dists):
        m, c, d = load(f, a, n, return_modifier=True, return_data=True, **kwargs)
        models.append(m)
        cmods.append(c)
        datas.append(d + m.config.auxdata)

    workspaces = []
    for m, c, d in zip(models, cmods, datas):
        if isinstance(c, Iterable):
            name = " ".join([cmod.name for cmod in c])
        else:
            name = c.name
        workspaces.append(_UnvalidatedWorkspace.build(m, d, name, validate=False))

    comb_ws = None
    for w in workspaces:
        if comb_ws:
            comb_ws = _UnvalidatedWorkspace.combine(comb_ws, w)
        else:
            comb_ws = w

    modifier_set = None
    for c in list(_flatten(cmods)):
        if modifier_set:
            modifier_set = modifier_set | c.expanded_pyhf
        else:
            modifier_set = c.expanded_pyhf

    model = pyhf.Model(
        comb_ws, validate=False, batch_size=None, modifier_set=modifier_set, **kwargs
    )

    if return_data:
        return model, comb_ws.data(model)
    return model


def _read_pars(json_input):
    """
    Parse lists to tuples for pyhf.
    """
    new_pars = deepcopy(json_input)
    for k, v in json_input.items():
        new_pars[k]["inits"] = tuple(v["inits"])
        new_pars[k]["bounds"] = tuple(tuple(w) for w in v["bounds"])
    return new_pars


def map(target_samples, kinematic_samples, target_bins, kinematic_bins, weights=None):
    """
    Generate mapping distribution from samples.
    Args:
        target_samples (array): Target (fitting variable) samples.
        kinematic_samples (array): Kinematic samples.
        target_bins (array): Target (fitting variable) binning.
        kinematic_bins (array): Kinematic binning.
        weights (array, optional): Weights for individual samples.
    """
    samples = [target_samples] + list(kinematic_samples)
    binning = [target_bins] + list(kinematic_bins)
    return np.histogramdd(samples, bins=binning, weights=weights)[0]


def _flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten(x)
        else:
            yield x
