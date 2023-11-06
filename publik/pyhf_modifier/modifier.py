from publik.pyhf_modifier import custom_modifier

import re
from collections import defaultdict
import numpy as np
import scipy as sp
import pyhf

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

    model = pyhf.Model(spec, validate=False, batch_size = None, modifier_set=modifier_set)

    return model

def add(new_pars, alt_dist, null_dist, map, binning):
    """
    Add modifier to the expanded modifier set for pyhf. This function adds the new modifier.
    """
    corr_pars, unco_pars = _separate_pars(new_pars)
    cmod = _reweight(alt_dist, null_dist, map, binning, corr_pars)
    weight_function= cmod.weight_func
    expanded_pyhf = custom_modifier.add('custom', list(unco_pars.keys()), unco_pars, namespace = {'weight_function': weight_function})
    return expanded_pyhf

def _separate_pars(new_pars):
    # find correlated pars
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

class _reweight():
    def __init__(self, alt_dist, null_dist, map, binning, corr_pars=None):
        self.alt_dist = alt_dist
        self.map = map
        self.binning = binning
        
        self.null_binned = bintegrate(null_dist, binning)
        
        # take care of correlated paramters
        self.corr_infos = {}
        if corr_pars:
            for k,v in corr_pars.items():
                self.corr_infos[k] = {
                    'mean': v['inits'],
                    'uvec': _pca(v['cov'])
                    }
                
        # cache previously called function values
        self.cache = {}
        
    def _rotate_pars(self, pars):
        """
        map from pca parameters to pyhf parameters
        """
        rot_pars = pars.copy()
        pyhf_shifts = defaultdict(list)

        for corr_k, corr_v in self.corr_infos.items():
            for par_k, par_v in pars.items():
                if corr_k == re.sub("_decorrelated[\(\[].*?[\)\]]", "", par_k):
                    pyhf_shifts[corr_k].append(par_v)

        for corr_k, pyhf_shift_list in pyhf_shifts.items():
            pyhf_shifts_arr = np.array(pyhf_shift_list)
            pars_shifts = corr_v['uvec'] @ pyhf_shifts_arr
            pars_new = corr_v['mean'] + pars_shifts
            for ind, par in enumerate(pars_new):
                rot_pars[corr_k + f'_decorrelated[{ind}]'] = par
        return rot_pars

    def get_weights(self, pars):
        """
        compute the new weights and process them for sensibility
        """
        
        # compute original parameters from pyhf parameters
        rot_pars = self._rotate_pars(pars)
        
        alt_binned = bintegrate(self.alt_dist, self.binning, tuple(rot_pars.values()))
                
        weights = alt_binned / self.null_binned
        
        weights[weights<0] = 1.
        weights[np.isnan(weights)] = 1.
        
        return weights
        
    def weight_func(self, pars):
        key = tuple(i for i in pars.items())
        if key in self.cache:
            return self.cache[key]
                                
        weights = self.get_weights(pars)
        
        def func(ibins):
            results = self.map @ (weights - 1)
            return np.array([results[i] for i in ibins])
        
        self.cache[key] = func
        
        return func
    
def bintegrate(func, bins, args=()):
    return np.array([sp.integrate.quad(func, q2min, q2max, args=args)[0] for q2min, q2max in zip(bins[:-1], bins[1:]) ]) 

def _pca(cov):
    """Principal Component analysis, moving to a space where the covariance matrix is diagonal
    https://www.cs.cmu.edu/~elaw/papers/pca.pdf

    Args:
        cov (array): Covariance matrix

    Returns:
        array: matrix of column wise error vectors (eigenvectors * sqrt(eigenvalues); sqrt(eigenvalues) = std)
    """
    svd = np.linalg.svd(cov)
    uvec = svd[0] @ np.sqrt(np.diag(svd[1]))
    return uvec