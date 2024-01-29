import numpy as np
from redist import modifier
import ksnunu_utils

from bayesian_pyhf import infer
from bayesian_pyhf import prepare_inference
import pymc as pm

null = ksnunu_utils.null_pred()
alt = ksnunu_utils.alt_pred()

model, alt_yields = modifier.load('ksnunu_model.json', alt.distribution, null.distribution, return_data=True, clip_bin_data=0.1)

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [3.], 'upper': [9.]},
    'cvr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [6.]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [6.]},
    'csr': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]},
    # 'csr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [3.]},
    'ctl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [6.]}
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)

# fix FF pars
for i in range(3, 19):
    ind = model.config.auxdata_order.index(f'FFKs_decorrelated[{i}]')
    model.constraint_model.constraints_gaussian.sigmas[ind] = 1e-10

n_draws = 10000
with infer.model(model, unconstr_priors, alt_yields):
    post_data = pm.sample(draws=n_draws, tune=1000, chains=8)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( '../samples/ksnunu_post_data.json')
post_pred.to_json( '../samples/ksnunu_post_pred.json')
prior_pred.to_json('../samples/ksnunu_prior_pred.json')
