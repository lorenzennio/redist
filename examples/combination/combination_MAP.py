import numpy as np
from redist import modifier

import sys, os
path2add = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
if (not (path2add in sys.path)) :
    sys.path.append(path2add)

from knunu import knunu_utils
from ksnunu import ksnunu_utils

import pymc as pm
from bayesian_pyhf import prepare_inference
from bayesian_pyhf import make_op

from contextlib import contextmanager

@contextmanager
def fixed_infer_model(stat_model, unconstrained_priors, data, ur_hyperparameters = None):
    '''
    Builds a context with the pyhf model set up as data-generating model. The priors for the constrained parameters
    have already been updated using conjugate priors.

    Args:
        - stat_model: pyhf model.
        - unconstrained_priors (dictionary): Dictionary of all unconstrained priors.
        - data (list or array): Observations used for the inference step.
    Returns:
        - model (context): Context in which PyMC methods can be used.
    '''
    priorDict = prepare_inference.build_priorDict(stat_model, unconstrained_priors, ur_hyperparameters)
    expData_op_Act = make_op.makeOp_Act(stat_model)

    with pm.Model() as m:
        pars = prepare_inference.priors2pymc(stat_model, priorDict)

        Expected_Data = pm.Poisson("Expected_Data", mu=expData_op_Act(pars), observed=data)
        # Expected_Data = pm.Normal("Expected_Data", mu=expData_op_Act(pars), observed=data)
        yield m


files       = ['../knunu/knunu_model.json', '../ksnunu/ksnunu_model.json']
alt_dists   = [knunu_utils.alt_pred().distribution, ksnunu_utils.alt_pred().distribution]
null_dists  = [knunu_utils.null_pred().distribution, ksnunu_utils.null_pred().distribution]

model, data = modifier.combine(files, alt_dists, null_dists, return_data=True, clip_bin_data=1e-6)
yields = data[:model.config.nmaindata]

# Perform the sampling
unconstr_priors = {
    'mu':  {'type': 'Normal_Unconstrained',  'mu':    [1.], 'sigma': [1e-10]},
    # 'cvl': {'type': 'Normal_Unconstrained',  'mu':    [6.6], 'sigma': [3]},
    # 'cvr': {'type': 'HalfNormal_Unconstrained',  'sigma': [3]},
    # 'csl': {'type': 'HalfNormal_Unconstrained',  'sigma': [3]},
    # 'csr': {'type': 'HalfNormal_Unconstrained',  'sigma': [3]},
    # 'ctl': {'type': 'HalfNormal_Unconstrained',  'sigma': [3]},
    'cvl': {'type': 'Uniform_Unconstrained', 'lower': [5.], 'upper': [15.]},
    'cvr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
    'csl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
    'csr': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
    'ctl': {'type': 'Uniform_Unconstrained', 'lower': [0.], 'upper': [10.]},
}

n_draws = 100000
with fixed_infer_model(model, unconstr_priors, yields) as m:

    # implement constraints due to exchange symmetry
    cvl = m.named_vars["cvl"]
    cvr = m.named_vars["cvr"]
    csl = m.named_vars["csl"]
    csr = m.named_vars["csr"]
    cv_constraint = cvl > cvr
    cs_constraint = csl > csr
    potential = pm.Potential("cv_constraint", pm.math.log(pm.math.switch(cv_constraint, 1, 0)))
    potential = pm.Potential("cs_constraint", pm.math.log(pm.math.switch(cs_constraint, 1, 0)))

    m = pm.find_MAP(start={'cvl': 10., 'cvr': 4., 'csl': 3., 'csr': 1., 'ctl': 1.}, maxeval=10000)
    # m = pm.find_MAP(start={'cvl': 7., 'cvr': 1., 'csl': 1., 'csr': 0.1, 'ctl': 0.1}, maxeval=50000)

for k,v in m.items():
    print(k,v)
