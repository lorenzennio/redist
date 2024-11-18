import pytest
import numpy as np
import pyhf
from redist import modifier

def null_dist(x, y, a=1., b=1.):
    return a*x**2 + b*y**2

def alt_dist(x, y, a=2., b=0.5):
    return null_dist(x, y, a, b)

class TestSimpleModel:
    xbinning = np.linspace( 0., 10., 5)
    ybinning = np.linspace( 4.,  8., 4)

    binning = [xbinning, ybinning]

    ones = np.ones((4,len(xbinning)-1,len(ybinning)-1))
    mapping_dist = np.array([m*(4-i) for i,m in enumerate(ones)])
    mapping_dist[0,:,1] = 2
    mapping_dist[1,:,2] = 4
    mapping_dist[2,:,0] = 1

     # Set up the custom modifier
    new_params = {
                    'a'   :{'inits': (1.,), 'bounds': ((0., 5.),), 'paramset_type': 'unconstrained'},
                    'b'   :{'inits': (1.,), 'bounds': ((0., 5.),), 'paramset_type': 'unconstrained'}
                }

    cmod = modifier.Modifier(new_params, alt_dist, null_dist, mapping_dist, binning)


    spec = {
    "channels": [
        {
        "name": "singlechannel",
        "samples": [
            {
            "name": "signal",
            "data": [np.sum(m) for m in mapping_dist],
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
            "data": [20*i**2 for i in range(len(mapping_dist))],
            "modifiers": [
                {
                "name": "uncorr_bkguncrt",
                "type": "shapesys",
                "data": [np.sqrt(20*i**2) for i in range(len(mapping_dist))]
                }
            ]
            }
        ]
        }
    ]
    }
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


    init = model.config.suggested_init()
    init[0] = 4.0
    init[1] = 1.0
    data = list(model.expected_actualdata(init)) + model.config.auxdata

    fixed = model.config.suggested_fixed()
    fixed[2] = True

    best_fit = pyhf.infer.mle.fit(data, model, fixed_params=fixed)

    def test_set_up_modifier(self):
        assert 'custom' in self.cmod.expanded_pyhf

    def test_add_custom_modifier(self):
        assert 'a' in self.model.config.par_map
        assert 'b' in self.model.config.par_map

    def test_yields(self):
        init = self.model.config.suggested_init()

        init[0] = 4.
        init[1] = 1.
        print(list(self.model.expected_actualdata(init)))
        assert list(self.model.expected_actualdata(init)) == [87.31293547591133, 106.2242274556989, 122.56775971774324, 206.14632294135367]

        init[0] = 3.
        init[1] = 2.
        assert list(self.model.expected_actualdata(init)) == [95.77097849197044, 115.40807581856629, 127.52258657258108, 208.71544098045123]

        init[0] = 1.5
        init[1] = 2.5
        assert list(self.model.expected_actualdata(init)) == [84.22902150802956, 104.59192418143371, 122.47741342741892, 205.28455901954877]

    def test_best_fit(self):
        assert pytest.approx(self.best_fit, 1e-4) == [4.00150799, 0.99904199, 1., 1., 0.99999687, 1.00000414, 1.00000162]
