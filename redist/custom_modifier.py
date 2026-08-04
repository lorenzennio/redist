import numexpr
import pyhf
import numpy as np
from pyhf.parameters import ParamViewer
from pyhf import get_backend
from pyhf import events
from typing import Any, Callable, Sequence


def add(funcname, par_names, newparams, input_set=None, namespace=None):
    namespace = namespace or {}

    def make_func(
        expression: str, namespace=namespace
    ) -> Callable[[Sequence[float]], Any]:
        def func(deps: Sequence[float]) -> Any:
            if expression in namespace:
                parvals = dict(zip(par_names, deps))
                return namespace[expression](parvals)()
            return numexpr.evaluate(
                expression, local_dict=dict(zip(par_names, deps), **namespace)
            )

        return func

    def _allocate_new_param(p):
        param_dict = {
            "paramset_type": p["paramset_type"]
            if "paramset_type" in p.keys()
            else "unconstrained",
            "n_parameters": 1,
            "is_shared": True,
            "inits": p["inits"],
            "bounds": p["bounds"],
            "is_scalar": True,
            "fixed": False,
            "auxdata": p["auxdata"] if "auxdata" in p.keys() else (0.0,),
        }
        return param_dict

    class _builder:
        is_shared = True

        def __init__(self, config):
            self.builder_data = {"funcs": {}}
            self.config = config
            self.required_parsets = {}

        def collect(self, thismod, nom):
            maskval = True if thismod else False
            mask = [maskval] * len(nom)
            return {"mask": mask}

        def append(self, key, channel, sample, thismod, defined_samp):
            self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
                "data", {"mask": []}
            )
            nom = (
                defined_samp["data"]
                if defined_samp
                else [0.0] * self.config.channel_nbins[channel]
            )
            moddata = self.collect(thismod, nom)
            self.builder_data[key][sample]["data"]["mask"] += moddata["mask"]
            if thismod:
                if thismod["name"] != funcname:
                    self.builder_data["funcs"].setdefault(
                        thismod["name"], thismod["data"]["expr"]
                    )
                self.required_parsets = {
                    k: [_allocate_new_param(v)] for k, v in newparams.items()
                }

        def finalize(self):
            return self.builder_data

    class _applier:
        name = funcname
        op_code = "multiplication"

        def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
            self.funcs = [make_func(f) for f in builder_data["funcs"].values()]

            self.batch_size = batch_size
            pars_for_applier = par_names
            _modnames = [f"{mtype}/{m}" for m, mtype in modifiers]

            parfield_shape = (
                (self.batch_size, pdfconfig.npars)
                if self.batch_size
                else (pdfconfig.npars,)
            )
            self.param_viewer = ParamViewer(
                parfield_shape, pdfconfig.par_map, pars_for_applier
            )
            self._custommod_mask = [
                [[builder_data[modname][s]["data"]["mask"]] for s in pdfconfig.samples]
                for modname in _modnames
            ]
            self._precompute()
            events.subscribe("tensorlib_changed")(self._precompute)

        def _precompute(self):
            tensorlib, _ = get_backend()
            if not self.param_viewer.index_selection:
                return
            self.custommod_mask = tensorlib.tile(
                tensorlib.astensor(self._custommod_mask),
                (1, 1, self.batch_size or 1, 1),
            )
            self.custommod_mask_bool = tensorlib.astensor(
                self.custommod_mask, dtype="bool"
            )
            self.custommod_default = tensorlib.ones(self.custommod_mask.shape)
            # backend-independent copy of the mask, used to build the scatter
            # indices below; depends only on the model layout, never on pars
            self._mask_np = np.tile(
                np.asarray(self._custommod_mask, dtype=bool),
                (1, 1, self.batch_size or 1, 1),
            )
            self._scatter_index_cache = {}

        def _scatter_indices(self, n_source):
            """Indices into a flat source that reproduce ``np.place``.

            ``np.place(target, mask, source)`` writes into the mask's True
            positions in C order, cycling through the flattened source when it
            is shorter. Gathering with these indices and then masking is the
            equivalent that works on every pyhf backend, unlike the in-place
            ``np.place``, which cannot be traced.

            Depends only on static shapes, so it is built once per source size.
            """
            indices = self._scatter_index_cache.get(n_source)
            if indices is None:
                flat_mask = self._mask_np.ravel()
                indices = np.zeros(flat_mask.size, dtype=int)
                indices[flat_mask] = np.arange(int(flat_mask.sum())) % n_source
                self._scatter_index_cache[n_source] = indices
            return indices

        def apply(self, pars):
            """
            Returns:
                modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
            """
            if not self.param_viewer.index_selection:
                return
            tensorlib, _ = get_backend()
            deps = self.param_viewer.get(pars)
            out = tensorlib.astensor([f(deps) for f in self.funcs])
            flat_out = tensorlib.ravel(out)
            indices = self._scatter_indices(int(tensorlib.shape(flat_out)[0]))
            results = tensorlib.reshape(
                tensorlib.gather(flat_out, indices),
                tensorlib.shape(self.custommod_mask),
            )
            # entries gathered outside the mask are discarded here, so they
            # contribute no value and no gradient
            return tensorlib.where(
                self.custommod_mask_bool, results, self.custommod_default
            )

    modifier_set = {_applier.name: (_builder, _applier)}
    modifier_set.update(
        **(input_set if input_set is not None else pyhf.modifiers.histfactory_set)
    )
    return modifier_set
