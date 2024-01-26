import numpy as np
from redist import modifier
import knunu_utils 
import ksnunu_utils 

from bayesian_pyhf import infer
from bayesian_pyhf import prepare_inference
import pymc as pm

files       = ['knunu_model.json', 'ksnunu_model.json']
alt_dists   = [knunu_utils.alt_pred().distribution, ksnunu_utils.alt_pred().distribution]
null_dists  = [knunu_utils.null_pred().distribution, ksnunu_utils.null_pred().distribution]

model, data = modifier.combine(files, alt_dists, null_dists, return_data=True, clip_bin_data=1e-6)
yields = data[:model.config.nmaindata]

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    # 'cvl': {'type': 'Normal_Unconstrained',  'mu':    [6.6], 'sigma':[5]},
    # 'cvr': {'type': 'HalfNormal_Unconstrained',  'sigma':[5]},
    # 'csl': {'type': 'HalfNormal_Unconstrained',  'sigma':[5]},
    # 'csr': {'type': 'HalfNormal_Unconstrained',  'sigma':[5]},
    # 'ctl': {'type': 'HalfNormal_Unconstrained',  'sigma':[5]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [5.], 'upper': [15.]},
    'cvr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
    'csr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
    'ctl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
}

with infer.model(model, unconstr_priors, yields):
    m = pm.find_MAP(start={'cvl': 10., 'cvr': 4., 'csl': 3., 'csr': 1., 'ctl': 1.}, maxeval=10000)
    # m = pm.find_MAP(maxeval=50000)
    
print(m)