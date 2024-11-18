import numpy as np
from redist import modifier
import knunu_utils

from bayesian_pyhf import infer
from bayesian_pyhf import prepare_inference
import pymc as pm

null = knunu_utils.null_pred()
alt = knunu_utils.alt_pred()

model, alt_yields = modifier.load('knunu_model.json', alt.distribution, null.distribution, return_data=True, clip_bin_data=1e-6)

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [5.], 'upper': [20.]},
    'cvr': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [15.]},
    'csr': {'type': 'Normal_Unconstrained',  'mu':    [0.], 'sigma': [1e-10]},
    'ctl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [15.]}
}

priorDict_conjugate = prepare_inference.build_priorDict(model, unconstr_priors)

n_draws = 100000
with infer.model(model, unconstr_priors, alt_yields):
    post_data = pm.sample(draws=n_draws,
                          tune=10000,
                          cores=8,
                          initvals={'cvl': 14.,
                                    'csl': 4.,
                                    'ctl': 1.}
                          )
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(n_draws)

post_data.to_json( '../samples_multiply/knunu_large_post_data.json')
post_pred.to_json( '../samples_multiply/knunu_large_post_pred.json')
prior_pred.to_json('../samples_multiply/knunu_large_prior_pred.json')
