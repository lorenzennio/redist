from publik.pyhf_modifier import custom_modifier

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

def add(new_params, alt_dist, null_dist, map, binning):
    """
    Add modifier to the expanded modifier set for pyhf. This function adds the new modifier.
    """
    cmod = reweight(alt_dist, null_dist, map, binning)
    weight_function= cmod.weight_func
    expanded_pyhf = custom_modifier.add('custom', list(new_params.keys()), new_params, namespace = {'weight_function': weight_function})
    return expanded_pyhf

class reweight():
    def __init__(self, alt_dist, null_dist, map, binning):
        self.alt_dist = alt_dist
        self.map = map
        self.binning = binning
        
        self.null_binned = bintegrate(null_dist, binning)
        
        # cache previously called function values
        self.cache = {}
        
    def get_weights(self, pars):
        """
        compute the new weights and process them for sensibility
        """
        alt_binned = bintegrate(self.alt_dist, self.binning, tuple(pars.values()))
                
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