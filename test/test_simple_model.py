import os 
import pytest
import numpy as np
import pyhf
import json
from redist import modifier

dir_path = os.path.dirname(__file__)

def null_dist(x, a=10.):
    return a

def alt_dist(x, a=1., h1=1., h2=1.):
    return a*(1+x*h1+x**2*h2)

class TestSimpleModel:
    binning = np.array([2,3,5,6])

    mapping_dist = np.array([[2., 2., 1.], [2., 6., 2.]])

    new_params = {
                'a'   :{'inits': (1.,), 'bounds': ((0., 10.),), 'paramset_type': 'unconstrained'},
                'h'   :{'inits': (1.,1.), 'bounds': (), 'cov': [[0.5,0.1],[0.1,0.5]], 'paramset_type': 'constrained_by_normal'}
            }

    cmod = modifier.Modifier(new_params, alt_dist, null_dist, mapping_dist, [binning])
    
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
        
        init[0] = 2.
        init[1] = -0.2
        init[2] = -0.2
        assert list(self.model.expected_actualdata(init)) == [70.87379321155956, 106.5473859310839]
        
        init[0] = 4.
        init[1] = -1.
        init[2] = 2.
        assert list(self.model.expected_actualdata(init)) == [130.75011810022602, 241.82138862829896]
        
        init[0] = 10.
        init[1] = -5.
        init[2] = 5.
        assert list(self.model.expected_actualdata(init)) == [534.8812568724203, 1153.7637634754315]

    def test_best_fit(self):
        print(self.best_fit)
        assert pytest.approx(self.best_fit, 1e-4) == [ 1.01373882e+00, -9.85076153e-04,  1.29777515e-03,  1.00000000e+00, 9.87905813e-01,  1.02699001e+00]