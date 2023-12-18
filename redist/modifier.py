import re
from copy import deepcopy
import itertools
from collections import defaultdict
import numpy as np
import scipy as sp
import json
import pyhf
from redist import custom_modifier

class Modifier():
    """
    Modifier implementation to reweight historgram according to the ratio of 
    a null and an alternative distribution. 
    """
    def __init__(self, new_pars, alt_dist, null_dist, map, bins, name = None):
        # store name
        self.name = name if name else 'custom'
        
        # store null and alternative distributions
        self.null_dist = null_dist
        self.alt_dist  = alt_dist
        
        # stor mapping distribution and binning
        shape = np.shape(map)
        self.map  = np.reshape(map, (shape[0], np.prod(shape[1:])))
        self.bins = bins
        
        # compute the bin-integrated null distribution (this is fixed)
        self.null_binned = bintegrate(null_dist, bins)
        
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
            namespace = {self.name + '_weight_fn': self.weight_func}
            )
        
    def _separate_pars(self, new_pars):
        """
        Separate parameters into correlated and uncorrelated ones.
        """
        corr_pars = {}
        unco_pars = {}
        for k,v in new_pars.items():
            if 'cov' in v.keys():
                corr_pars[k] = v
                # for each correlated parameter, add one pyhf parameter
                for n, _ in enumerate(v['inits']):
                    name = k + f'_decorrelated[{n}]'
                    unco_pars[name] = {
                        'inits': (0.0,),
                        'bounds': ((-5.0, 5.0),),
                        'paramset_type': v['paramset_type']
                        }
            else:
                unco_pars[k] = v

        return corr_pars, unco_pars
                
    def _corr_infos(self, corr_pars):
        """
        Compute and store pca rotation matrix for correlated parameters.
        """
        corr_infos = {}
        if corr_pars:
            for k,v in corr_pars.items():
                corr_infos[k] = {
                    'mean': v['inits'],
                    'uvec': _pca(v['cov'])
                    }

        return corr_infos
        
    def rotate_pars(self, pars):
        """
        Map from pca parameters to true parameters.
        """
        rot_pars = {}
        for k,v in pars.items():
            rot_pars[re.sub('_decorrelated', '', k)] = v
        pyhf_shifts = defaultdict(list)

        for corr_k, corr_v in self.corr_infos.items():
            for par_k, par_v in pars.items():
                if corr_k == re.sub('_decorrelated[\(\[].*?[\)\]]', '', par_k):
                    pyhf_shifts[corr_k].append(par_v)

        for corr_k, pyhf_shift_list in pyhf_shifts.items():
            pyhf_shifts_arr = np.array(pyhf_shift_list)
            pars_shifts = corr_v['uvec'] @ pyhf_shifts_arr
            pars_new = corr_v['mean'] + pars_shifts
            for ind, par in enumerate(pars_new):
                rot_pars[corr_k + f'[{ind}]'] = par
        
        return rot_pars

    def get_weights(self, pars):
        """
        Compute the new weights and process them for sensibility.
        """
        # compute original parameters from pyhf parameters
        rot_pars = self.rotate_pars(pars)
        
        alt_binned = bintegrate(self.alt_dist, self.bins, tuple(rot_pars.values()))
                
        weights = alt_binned / self.null_binned
        
        weights[weights<0] = 1.
        weights[np.isnan(weights)] = 1.
        
        #flatten the weights
        weights = weights.reshape(-1, order='F')

        return weights
        
    def weight_func(self, pars):
        """
        Build function that applies weights to histogram.
        """
        key = tuple(i for i in pars.items())
        if key in self.cache:
            return self.cache[key]
                                
        weights = self.get_weights(pars)
        results = self.map @ (weights - 1)
        
        def func():
            return results
        
        self.cache[key] = func
        
        return func
    
def bintegrate(func, bins, args=()):
    """
    Integrate function in given bins.
    """
    ranges = [list(zip(b[:-1], b[1:])) for b in bins]
    results = []
    for limits in itertools.product(*ranges):
        results.append(sp.integrate.nquad(func, limits, args=args)[0])
    return np.reshape(results, tuple(len(b)-1 for b in bins)).T
    
def _pca(cov, return_rot=False):
    """Principal Component analysis, moving to a space where the covariance matrix is diagonal
    https://www.cs.cmu.edu/~elaw/papers/pca.pdf

    Args:
        cov (array): Covariance matrix

    Returns:
        array: matrix of column wise error vectors (eigenvectors * sqrt(eigenvalues); sqrt(eigenvalues) = std)
    """
    svd = np.linalg.svd(cov)
    uvec = svd[0] @ np.sqrt(np.diag(svd[1]))
    if return_rot:
        return uvec, svd[0]
    return uvec

def par_dict(model, pars):
    """
    Build parmaeter dictionary for pyhf model.
    """
    return {k: pars[v['slice']][0] if len(pars[v['slice']])==1 else pars[v['slice']].tolist() for k, v in model.config.par_map.items()}

def add_to_model(model, channels, samples, modifier_set, modifier_specs):
    """
    Add a custom modifier to a pyhf model.
    """
    spec = model.spec
    
    for c, chan in enumerate(spec['channels']):
        if chan['name'] in channels:
            for s, samp in enumerate(chan['samples']):
                if samp['name'] in samples:
                  spec['channels'][c]['samples'][s]['modifiers'].append(modifier_specs)

    model = pyhf.Model(spec, validate=False, batch_size=None, modifier_set=modifier_set)

    return model

def save(file, spec, cmod, data=None):
    """
    Save the custom model, mapping distribution (and data).
    """
    d = {
        'spec': spec, 
        'name': cmod.name,
        'new_pars': cmod.new_pars,
        'map':  cmod.map.tolist(),
        'bins': cmod.bins if isinstance(cmod.bins,list) else cmod.bins.tolist()
        }
    if data is not None:
        d['data'] = np.array(data).tolist()
    
    with open(file, 'w') as f:
        json.dump(d, f, indent=4)
        
def load(file, alt_dist, null_dist, return_modifier=False, return_data=False, **kwargs):
    """
    Load and build model from file
    """
    with open(file, 'r') as f:
        d = json.load(f)
        
    new_pars = _read_pars(d['new_pars'])
    
    cmod = Modifier(new_pars, alt_dist, null_dist, d['map'], d['bins'], name=d['name'])
    
    model = pyhf.Model(d['spec'], validate=False, batch_size=None, modifier_set=cmod.expanded_pyhf, **kwargs)
        
    if return_modifier and return_data: return model, cmod, d['data']
    if return_modifier: return model, cmod
    if return_data: return model, d['data']
    return model

def combine(files, alt_dists, null_dists, return_data=False, **kwargs):
    models = []
    cmods  = []
    datas  = []
    for f, a, n in zip(files, alt_dists, null_dists):
        m, c, d = load(f, a, n, return_modifier=True, return_data=True, **kwargs)
        models.append(m)
        cmods.append(c)
        datas.append(d + m.config.auxdata)
    
    workspaces = []
    for m, c, d in zip(models, cmods, datas):
        workspaces.append(pyhf.Workspace.build(m, d, c.name, validate=False))
    
    comb_ws = None
    for w in workspaces:
        if comb_ws:
            comb_ws = pyhf.Workspace.combine(comb_ws, w, validate=False)
        else:
            comb_ws = w
    
    modifier_set = None
    for c in cmods:
        if modifier_set:
            modifier_set = modifier_set | c.expanded_pyhf
        else:
            modifier_set = c.expanded_pyhf
    
    model = pyhf.Model(comb_ws, validate=False, batch_size=None, modifier_set=modifier_set, **kwargs)

    if return_data: return model, comb_ws.data(model)
    return model
    
def _read_pars(json_input):
    """
    Parse lists to tuples for pyhf.
    """
    new_pars = deepcopy(json_input)
    for k, v in json_input.items():
        new_pars[k]['inits'] = tuple(v['inits'])
        new_pars[k]['bounds'] = tuple(tuple(w) for w in v['bounds'])
    return new_pars

def map(target_samples, kinematic_samples, target_bins, kinematic_bins):
    """
    Generate mapping distribution from samples.
    """
    samples = [target_samples] + list(kinematic_samples)
    binning = [target_bins] + list(kinematic_bins)
    return np.histogramdd(samples, bins=binning)[0]