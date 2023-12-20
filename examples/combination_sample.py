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

model, data = modifier.combine(files, alt_dists, null_dists, return_data=True, clip_bin_data=0.1)
yields = data[:model.config.nmaindata]

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [3.], 'upper': [7.]},
    'cvr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [4.]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [4.]},
    'csr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [4.]},
    'ctl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [4.]},
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)

# fix FF pars
for i in range(3, 19):
    ind = model.config.auxdata_order.index(f'FFKs_decorrelated[{i}]')
    model.constraint_model.constraints_gaussian.sigmas[ind] = 1e-10

for i in range(3, 8):
    ind = model.config.auxdata_order.index(f'FFK_decorrelated[{i}]')
    model.constraint_model.constraints_gaussian.sigmas[ind] = 1e-10

n_draws = 100000
with infer.model(model, unconstr_priors, yields):
    post_data = pm.sample(draws=n_draws, tune=10000, cores=8)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( 'samples/comb_post_data.json')
post_pred.to_json( 'samples/comb_post_pred.json')
prior_pred.to_json('samples/comb_prior_pred.json')
