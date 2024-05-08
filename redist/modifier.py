import re
from copy import deepcopy
import itertools
from collections import defaultdict
from collections.abc import Iterable
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
    def __init__(self, new_pars, alt_dist, null_dist, map, bins, name = None, cutoff=None, weight_bound=None):
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
        """        
        # store name and cutoff
        self.name = name if name else 'custom'
        self.cutoff = cutoff
        self.weight_bound = weight_bound
        
        # store null and alternative distributions
        self.null_dist = null_dist
        self.alt_dist  = alt_dist
        
        # stor mapping distribution and binning
        shape = np.shape(map)
        self.map  = np.reshape(map, (shape[0], np.prod(shape[1:])))
        self.bins = bins
        
        self.nominal = np.sum(self.map, axis=1)
        
        # compute the bin-integrated null distribution (this is fixed)
        self.null_binned = bintegrate(null_dist, bins, cutoff=self.cutoff)
        
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
        
        Args:
            new_pars (dict): New parameters to parametrize the model.
            
        Returns:
            dict, dict: Correlated and uncorrelated parameters.
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
        Compute and store svd rotation matrix for correlated parameters.
        
        Args:
            corr_pars (dict): Subset of `new_pars` containing correlated parameters.
            
        Returns:
            dict: Dictionary containing the mean and rotation matrix for each correlated parameter.
        """
        corr_infos = {}
        if corr_pars:
            for k,v in corr_pars.items():
                corr_infos[k] = {
                    'mean': v['inits'],
                    'uvec': _svd(v['cov'])
                    }

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
        
        Args:
            pars (dict): pyhf parameters.
            
        Returns:
            array: Weights for the given parameters.
        """
        # compute original parameters from pyhf parameters
        rot_pars = self.rotate_pars(pars)
        
        alt_binned = bintegrate(self.alt_dist, self.bins, tuple(rot_pars.values()), cutoff=self.cutoff)
                
        weights = alt_binned / self.null_binned
        
        weights[np.isnan(weights)] = 1.
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
        results = self.map @ weights
        results = results / self.nominal
        
        def func():
            return results
        
        self.cache[key] = func
        
        return func
    
def bintegrate(func, bins, args=(), cutoff=None):
    """
    Integrate function in given bins.
    
    Args:
        func (callable): Function to be integrated.
        bins (array): Binning of the integration.
        args (tuple, optional): Additional arguments for the function. Defaults to ().
        cutoff (tuple, optional): Cutoff values for the integration. Defaults to None.

    Returns:
        _type_: _description_
    """
    cutoff = cutoff if cutoff else tuple((-np.inf, np.inf) for _ in bins)
    ranges = [list(zip(b[:-1], b[1:])) for b in bins]
    results = []
    for limits in itertools.product(*ranges):
        #enforce cutoff
        if any(l[0] < c[0] or l[1] > c[1] for l, c in zip(limits, cutoff)):
            results.append(np.nan)
        else:
            results.append(sp.integrate.nquad(func, limits, args=args)[0])
    return np.reshape(results, tuple(len(b)-1 for b in bins)).T
    
def _svd(cov, return_rot=False):
    """Singular value decomposition, moving to a space where the covariance matrix is diagonal
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
    
    Args:
        model (pyhf.Model): pyhf model.
        pars (dict): Parameters.
        
    Returns:
        dict: Dictionary of parameters by names.
    """
    return {k: pars[v['slice']][0] if len(pars[v['slice']])==1 else pars[v['slice']].tolist() for k, v in model.config.par_map.items()}

def add_to_model(model, channels, samples, modifier_set, modifier_specs, **model_kwargs):
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
    
    for c, chan in enumerate(spec['channels']):
        if chan['name'] in channels:
            for s, samp in enumerate(chan['samples']):
                if samp['name'] in samples:
                    spec['channels'][c]['samples'][s]['modifiers'].append(modifier_specs)

    model = pyhf.Model(spec, validate=False, batch_size=None, modifier_set=modifier_set, **model_kwargs)

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
        'spec'          : spec, 
        'name'          : [cmod.name for cmod in cmods],
        'new_pars'      : [cmod.new_pars for cmod in cmods],
        'map'           : [cmod.map.tolist() for cmod in cmods],
        'bins'          : [cmod.bins if isinstance(cmod.bins,list) else cmod.bins.tolist() for cmod in cmods],
        'cutoff'        : [cmod.cutoff for cmod in cmods], 
        'weight_bound'  : [cmod.weight_bound for cmod in cmods]
        }
    if data is not None:
        d['data'] = np.array(data).tolist()
    
    with open(file, 'w') as f:
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
    with open(file, 'r') as f:
        d = json.load(f)
        
    new_pars = {}
    for pars in d['new_pars']:
        new_pars.update(_read_pars(pars))
    cmods = []
    for name, map, bins, cutoff, weight_bound in zip(d['name'], d['map'], d['bins'], d['cutoff'], d['weight_bound']):
        cmods.append(Modifier(new_pars, alt_dist, null_dist, map, bins, 
                              name=name, cutoff=cutoff, weight_bound=weight_bound))

    expanded_pyhf = {}
    for cmod in cmods:
        expanded_pyhf.update(cmod.expanded_pyhf)
        
    model = pyhf.Model(d['spec'], validate=False, batch_size=None, modifier_set=expanded_pyhf, **kwargs)
        
    if return_modifier and return_data: return model, cmods, d['data']
    if return_modifier: return model, cmods
    if return_data: return model, d['data']
    return model

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
    cmods  = []
    datas  = []
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
        workspaces.append(pyhf.Workspace.build(m, d, name, validate=False))
    
    comb_ws = None
    for w in workspaces:
        if comb_ws:
            comb_ws = pyhf.Workspace.combine(comb_ws, w, validate=False)
        else:
            comb_ws = w
    
    modifier_set = None
    for c in list(_flatten(cmods)):
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
    
    Args:
        target_samples (array): Target (fitting variable) samples.
        kinematic_samples (array): Kinematic samples.
        target_bins (array): Target (fitting variable) binning.
        kinematic_bins (array): Kinematic binning.
    """
    samples = [target_samples] + list(kinematic_samples)
    binning = [target_bins] + list(kinematic_bins)
    return np.histogramdd(samples, bins=binning)[0]

def _flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten(x)
        else:
            yield x