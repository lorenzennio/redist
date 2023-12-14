import numpy as np
from redist import modifier
import knunu_utils 
import ksnunu_utils 

from bayesian_pyhf import infer
from bayesian_pyhf import prepare_inference
import pymc as pm

files       = ['knunu_model_1b.json', 'ksnunu_model_1b.json']
alt_dists   = [knunu_utils.alt_pred().distribution, ksnunu_utils.alt_pred().distribution]
null_dists  = [knunu_utils.null_pred().distribution, ksnunu_utils.null_pred().distribution]

model, data = modifier.combine(files, alt_dists, null_dists, return_data=True)
yields = data[:model.config.nmaindata]

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [3.], 'upper': [9.]},
    'cvr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [6.]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [6.]},
    'csr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [6.]},
    'ctl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [6.]}
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)
priorDict_conjugate

n_draws = 100000
with infer.model(model, unconstr_priors, yields):
    post_data = pm.sample(draws=n_draws, tune=10000)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( 'samples/comb_1b_post_data.json')
post_pred.to_json( 'samples/comb_1b_post_pred.json')
prior_pred.to_json('samples/comb_1b_prior_pred.json')
