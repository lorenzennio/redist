import numpy as np
from redist import modifier
import pilnu_utils

from bayesian_pyhf import infer
from bayesian_pyhf import prepare_inference
import pymc as pm

null = pilnu_utils.null_pred()
alt = pilnu_utils.alt_pred()

model, alt_yields = modifier.load('pilnu_model.json', alt.distribution, null.distribution, return_data=True, clip_bin_data=0.1)

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.0], 'sigma': [1e-10]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [0.0], 'upper': [2.]},
    'cvr': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.0], 'upper': [2.]},
    'csr': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]},
    'ctl': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]}
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)

n_draws = 1000
with infer.model(model, unconstr_priors, alt_yields):
    post_data = pm.sample(draws=n_draws, tune=100)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( '../samples/pilnu_post_data.json')
post_pred.to_json( '../samples/pilnu_post_pred.json')
prior_pred.to_json('../samples/pilnu_prior_pred.json')
