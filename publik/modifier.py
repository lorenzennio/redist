import re
from collections import defaultdict
import numpy as np
import scipy as sp
import pyhf
from publik import custom_modifier

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

class Modifier():
    """
    Modifier implementation to reweight historgram according to the ratio of 
    a null and an alternative distribution. 
    """
    def __init__(self, new_pars, alt_dist, null_dist, map, bins):
        # store null and alternative distributions
        self.null_dist = null_dist
        self.alt_dist = alt_dist
        
        # stor mapping distribution and binning
        self.map = map
        self.bins = bins
        
        # compute the bin-integrated null distribution (this is fixed)
        self.null_binned = bintegrate(null_dist, bins)
        
        # take care of correlated paramters
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
            'custom', 
            list(self.unco_pars.keys()), 
            self.unco_pars, 
            namespace = {'weight_function': self.weight_func}
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
        
        return weights
        
    def weight_func(self, pars):
        """
        Build function that applies weights to histogram.
        """
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
    """
    Integrate function in given bins.
    """
    return np.array([sp.integrate.quad(func, q2min, q2max, args=args)[0] for q2min, q2max in zip(bins[:-1], bins[1:]) ]) 

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