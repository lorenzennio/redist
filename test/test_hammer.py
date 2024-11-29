import os
import pytest
import numpy as np
import pyhf
import json
from redist import hammer_fit
from redist import modifier

dir_path = os.path.dirname(__file__)

def null_dist(a=10.):
    return np.array([a,a])

def alt_dist(a=1., **kwargs):
    return np.array([a*(1+kwargs["h[0]"]+kwargs["h[1]"]),a*(1+3*kwargs["h[0]"]+9*kwargs["h[1]"])])

# Test class for Hammer_Modifier
class TestHammerModifier:
    new_params = {
            'a'   :{'inits': (1.,), 'bounds': ((0., 10.),), 'paramset_type': 'unconstrained'},
            'h'   :{'inits': (1.,1.), 'bounds': (), 'cov': [[0.5,0.1],[0.1,0.5]], 'paramset_type': 'constrained_by_normal'}
        }
    cmod = hammer_fit.Hammer_Modifier(new_params, alt_dist, null_dist)
    
    file = dir_path + "/models/simple_model.json"

    with open(file, 'r') as f:
        spec = json.load(f)

    model = pyhf.Model(spec)

    custom_mod = {
                "name": "theory",
                "type": "custom",
                "data":
                    {
                        "expr": "custom_weight_fn",
                    }
              }
    
    model = modifier.add_to_model(model, ['singlechannel'], ['signal'], cmod.expanded_pyhf, custom_mod)
    data = [58., 85.] + model.config.auxdata

    fixed = model.config.suggested_fixed()
    fixed[3] = True

    best_fit = pyhf.infer.mle.fit(data, model, fixed_params=fixed)

    def test_set_up_modifier(self):
        assert 'custom' in self.cmod.expanded_pyhf

    def test_add_custom_modifier(self):
        assert 'h_decorrelated[0]' in self.model.config.par_map
        assert 'h_decorrelated[1]' in self.model.config.par_map

    def test_yields(self):
        init = self.model.config.suggested_init()

        init[0] = 4.
        init[1] = -1.
        init[2] = 2.
        assert pytest.approx(list(self.model.expected_actualdata(init)), 1e-8) == [58.19089023, 159.75693534]

        init[0] = 10.
        init[1] = -5.
        init[2] = 5.
        assert pytest.approx(list(self.model.expected_actualdata(init)), 1e-8) == [92.38612788, 652.79761315]

    def test_best_fit(self):
        assert pytest.approx(self.best_fit, 1e-4) == [ 2.09390946 ,0.02796296,-0.03985101 ,1.,1.03135326 ,0.98326032]
