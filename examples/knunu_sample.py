import numpy as np
from redist import modifier
import knunu_utils

from bayesian_pyhf import infer
from bayesian_pyhf import prepare_inference
import pymc as pm

null = knunu_utils.null_pred()
alt = knunu_utils.alt_pred()

model, alt_yields = modifier.load('knunu_model.json', alt.distribution, null.distribution, return_data=True, clip_bin_data=5.)

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    'cvl': {'type': 'Normal_Unconstrained',  'mu':    [6.6], 'sigma': [10]},
    'cvr': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]},
    'csl': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [10]},
    'csr': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]},
    'ctl': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [10]},
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)
priorDict_conjugate

n_draws = 10000
with infer.model(model, unconstr_priors, alt_yields):
    post_data = pm.sample(draws=n_draws, tune=2000)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( 'samples/knunu_post_data.json')
post_pred.to_json( 'samples/knunu_post_pred.json')
prior_pred.to_json('samples/knunu_prior_pred.json')
