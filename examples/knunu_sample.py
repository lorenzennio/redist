import numpy as np
from publik import modifier
import knunu_utils

from Bayesian_pyhf import infer
from Bayesian_pyhf import prepare_inference
import pymc as pm

null = knunu_utils.null_pred()
alt = knunu_utils.alt_pred()

model, alt_yields = modifier.load('knunu_model.json', alt.distribution, null.distribution, return_data=True)

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained', 'mu': [1.], 'sigma': [1e-10]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [2.], 'upper': [10.]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [5.]},
    'ctl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [5.]}
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)
priorDict_conjugate

n_draws = 10000
with infer.model(model, unconstr_priors, alt_yields):
    # step = pm.Metropolis()
    post_data = pm.sample(draws=n_draws)#, step=step, tune=1000)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( 'samples/nuts_post_data.json')
post_pred.to_json( 'samples/nuts_post_pred.json')
prior_pred.to_json('samples/nuts_prior_pred.json')
