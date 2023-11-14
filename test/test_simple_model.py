import pytest
import numpy as np
import pyhf
from publik import modifier

def null_dist(x, a=1., h1=2., h2=0.):
    return a*(1+x*h1-x**2*h2)

def alt_dist(x, a=1., h1=1., h2=1.):
    return a*(1-x*h1+x**2*h2)

class TestSimpleModel:
    binning = np.array([2.,3.,4.])
    map = np.array([[5.,0], [0,10.]])


    spec = {
        "channels": [
            {
            "name": "singlechannel",
            "samples": [
                {
                "name": "signal",
                "data": [5.0, 10.0],
                "modifiers": [
                    {
                        "name": "mu",
                        "type": "normfactor",
                        "data": None
                    }
                ]
                },
                {
                "name": "background",
                "data": [50.0, 60.0],
                "modifiers": [
                    {
                    "name": "uncorr_bkguncrt",
                    "type": "shapesys",
                    "data": [5.0, 12.0]
                    }
                ]
                }
            ]
            }
        ]
        }
    model = pyhf.Model(spec)

    new_params = {
                    'a'   :{'inits': (1.,), 'bounds': ((0., 10.),), 'paramset_type': 'unconstrained'},
                    'h'   :{'inits': (1.,1.), 'bounds': ((0., 5.),(1., 6.)), 'cov': [[1.,0.5],[0.5,1.]], 'paramset_type': 'constrained_by_normal'}
                }

    cmod = modifier.Modifier(new_params, alt_dist, null_dist, map, binning)
    
    custom_mod = {
                "name": "theory",
                "type": "custom",
                "data":
                    {
                        "expr": "weight_function",
                        "ibin": [0, 1]
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
        assert list(self.model.expected_actualdata(init)) == [57.68992134928011, 84.44994553338127]
        
        init[0] = 4.
        init[1] = -1.
        init[2] = 2.
        assert list(self.model.expected_actualdata(init)) == [106.62143571502338, 226.58278866714602]
        
        init[0] = 10.
        init[1] = -5.
        init[2] = 5.
        assert list(self.model.expected_actualdata(init)) == [412.62905754890335, 1155.8265250059922]

    def test_best_fit(self):
        print(self.best_fit)
        assert pytest.approx(self.best_fit, 1e-4) == [ 2.0232e+00, -1.3993e-03, -1.5587e-03,  1.0000e+00, 9.9912e-01,  1.0014e+00]