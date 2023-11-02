import numexpr
import pyhf
from pyhf.parameters import ParamViewer
from pyhf import get_backend
from pyhf import events

def add(funcname, deps, newparams, input_set = None, namespace={}):
    globals().update(namespace)
    def make_func(expression,ibin):
        def func(dep_par_array):
            if expression in globals():
                parvals = dict(zip(deps,dep_par_array))
                to_integrate = globals()[expression](parvals)
            else:
                def to_integrate(xi):
                    parvals = dict(zip(deps,dep_par_array))
                    parvals['x'] = xi
                    return numexpr.evaluate(
                        expression,
                        global_dict=parvals
                    )
            probs = to_integrate(ibin)
            return probs
        return func

    def _allocate_new_param(p):
        param_dict = {
            'paramset_type': 'unconstrained',
            'n_parameters': 1,
            'is_shared': True,
            'inits': p['inits'],
            'bounds': p['bounds'],
            'is_scalar': True,
            'fixed': False,
        }
        try:
            if p['paramset_type'] == 'constrained_by_normal':
                param_dict['paramset_type'] = 'constrained_by_normal'
                param_dict['auxdata'] = (0.0,)
                if p['auxdata']:
                    param_dict['auxdata'] = p['auxdata']
        except:
            pass
        return param_dict
    
    class _builder:
        is_shared = True

        def __init__(self, config):
            self.builder_data = {'funcs': {}}
            self.config = config
            self.required_parsets = {}

        def collect(self, thismod, nom):
            maskval = True if thismod else False
            mask = [maskval] * len(nom)
            return {'mask': mask}

        def append(self, key, channel, sample, thismod, defined_samp):
            self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
                'data', {'mask': []}
            )
            nom = (
                defined_samp['data']
                if defined_samp
                else [0.0] * self.config.channel_nbins[channel]
            )
            moddata = self.collect(thismod, nom)
            self.builder_data[key][sample]['data']['mask'] += moddata['mask']
            if thismod:
                if thismod['name'] != funcname:
                    self.builder_data['funcs'].setdefault(thismod['name'],[thismod['data']['expr'],thismod['data']['ibin']])
                self.required_parsets = {k:[_allocate_new_param(v)] for k,v in newparams.items()}

        def finalize(self):
            return self.builder_data

    class _applier:
        name = funcname
        op_code = 'addition'

        def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
            self.funcs = [make_func(f,i) for f,i in builder_data['funcs'].values()]
            
            self.batch_size = batch_size
            pars_for_applier = deps
            _modnames = [f'{mtype}/{m}' for m, mtype in modifiers]

            parfield_shape = (
                (self.batch_size, pdfconfig.npars)
                if self.batch_size
                else (pdfconfig.npars,)
            )
            self.param_viewer = ParamViewer(
                parfield_shape, pdfconfig.par_map, pars_for_applier
            )
            self._custommod_mask = [
                [[builder_data[modname][s]['data']['mask']] for s in pdfconfig.samples]
                for modname in _modnames
            ]
            self._precompute()
            events.subscribe('tensorlib_changed')(self._precompute)

        def _precompute(self):
            tensorlib, _ = get_backend()
            if not self.param_viewer.index_selection:
                return
            self.custommod_mask = tensorlib.tile(
                tensorlib.astensor(self._custommod_mask), (1, 1, self.batch_size or 1, 1)
            )
            self.custommod_mask_bool = tensorlib.astensor(
                self.custommod_mask, dtype="bool"
            )
            self.custommod_default = tensorlib.zeros(self.custommod_mask.shape)

        def apply(self, pars):
            """
            Returns:
                modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
            """
            if not self.param_viewer.index_selection:
                return
            tensorlib, _ = get_backend()
            if self.batch_size is None:
                deps = self.param_viewer.get(pars)
                results = tensorlib.astensor([f(deps) for f in self.funcs])
                results = tensorlib.einsum(
                    'msab,mb->msab', self.custommod_mask, results
                )
            else:
                deps = self.param_viewer.get(pars)
                results = tensorlib.astensor([f(deps) for f in self.funcs])
                results = tensorlib.einsum(
                    'msab,ma->msab', self.custommod_mask, results
                )
            results = tensorlib.where(
                self.custommod_mask_bool, results, self.custommod_default
            )
            return results
    
    modifier_set = {_applier.name: (_builder, _applier)}
    modifier_set.update(**(input_set if input_set is not None else pyhf.modifiers.histfactory_set))
    return modifier_set