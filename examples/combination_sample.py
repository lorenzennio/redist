import numpy as np
from redist import modifier
import knunu_utils 
import ksnunu_utils 

from Bayesian_pyhf import infer
from Bayesian_pyhf import prepare_inference
import pymc as pm

files       = ['knunu_model.json', 'ksnunu_model.json']
alt_dists   = [knunu_utils.alt_pred().distribution, ksnunu_utils.alt_pred().distribution]
null_dists  = [knunu_utils.null_pred().distribution, ksnunu_utils.null_pred().distribution]

model, data = modifier.combine(files, alt_dists, null_dists, return_data=True, clip_bin_data=5.)
yields = data[:model.config.nmaindata]

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    'cvl': {'type': 'Normal_Unconstrained',  'mu':    [6.], 'sigma': [5.]},
    'cvr': {'type': 'HalfNormal_Unconstrained',             'sigma': [5.]},
    'csl': {'type': 'HalfNormal_Unconstrained',             'sigma': [5.]},
    'csr': {'type': 'HalfNormal_Unconstrained',             'sigma': [5.]},
    'ctl': {'type': 'HalfNormal_Unconstrained',             'sigma': [5.]}
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)
priorDict_conjugate

n_draws = 10000
with infer.model(model, unconstr_priors, yields):
    post_data = pm.sample(draws=n_draws, tune=2000)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( 'samples/comb_normal_post_data.json')
post_pred.to_json( 'samples/comb_normal_post_pred.json')
prior_pred.to_json('samples/comb_normal_prior_pred.json')
