import numpy as np
import eos
import numbers
import yaml
from yaml.loader import SafeLoader

def analysis():
    """
    Specify the likelihoods and FF parameter ranges 
    
    Returns:
        EOS analysis instance
    """
    with open('ksnunu_constraint.yaml', 'r') as f:
        constr = yaml.load(f, Loader=SafeLoader)

    analysis_args = {
        'priors': [
            { 'parameter': 'B->K^*::alpha^V_0@BSZ2015'  , 'min': -10, 'max': 10, 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^V_1@BSZ2015'  , 'min': -10, 'max': 10, 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^V_2@BSZ2015'  , 'min': -10, 'max': 10, 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A1_0@BSZ2015' , 'min': -10, 'max': 10, 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A1_1@BSZ2015' , 'min': -10, 'max': 10, 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A1_2@BSZ2015' , 'min': -10, 'max': 10, 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A12_1@BSZ2015', 'min': -10, 'max': 10, 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A12_2@BSZ2015', 'min': -10, 'max': 10, 'type': 'uniform' }
        ],
        'manual_constraints': constr,
        'likelihood': [
        ]
    }

    analysis = eos.Analysis(**analysis_args)
    analysis.optimize()
    return analysis

def efficiency(q2):
    """
    Efficiency map adapted from https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.127.181802/suppl_mat.pdf (Figure 3)
    Overall scale different, as this can be compensated be number of simulated events.
    """
    return np.exp(-0.2*q2)

class null_pred:
    """
    Null (SM) prediction
    """
    def __init__(self):
        p = analysis().parameters
        k = eos.Kinematics({'q2': 0.})
        o = eos.Options(**{'form-factors': 'BSZ2015', 'model': 'WET'})
        
        self.kv1 = k['q2']

        self.obs = eos.Observable.make('B->K^*nunu::dBR/dq2', p, k, o)

    def distribution(self, q2):
        if isinstance(q2, numbers.Number):
            self.kv1.set(q2)
            obs = self.obs.evaluate()
        else:
            obs = []
            for q in q2:
                self.kv1.set(q)
                obs.append(self.obs.evaluate())

        return obs
    
class alt_pred:
    """
    Alternative (BSM) prediction
    """
    def __init__(self):
        self.ana = analysis()
        p = self.ana.parameters
        k = eos.Kinematics({'q2': 0.})
        o = eos.Options(**{'form-factors': 'BSZ2015', 'model': 'WET'})
        
        self.kv1 = k['q2'                         ]
        self.wc1 = p['sbnunu::Re{cVL}'            ]
        self.wc2 = p['sbnunu::Re{cVR}'            ]
        self.wc3 = p['sbnunu::Re{cSL}'            ]
        self.wc4 = p['sbnunu::Re{cSR}'            ]
        self.wc5 = p['sbnunu::Re{cTL}'            ]
        self.hv1 = p['B->K^*::alpha^V_0@BSZ2015'  ]
        self.hv2 = p['B->K^*::alpha^V_1@BSZ2015'  ]
        self.hv3 = p['B->K^*::alpha^V_2@BSZ2015'  ]
        self.hv4 = p['B->K^*::alpha^A1_0@BSZ2015' ]
        self.hv5 = p['B->K^*::alpha^A1_1@BSZ2015' ]
        self.hv6 = p['B->K^*::alpha^A1_2@BSZ2015' ]
        self.hv7 = p['B->K^*::alpha^A12_1@BSZ2015']
        self.hv8 = p['B->K^*::alpha^A12_2@BSZ2015']
        
        self.obs = eos.Observable.make('B->K^*nunu::dBR/dq2', p, k, o)

    def distribution(self, q2, cvl, cvr, csl, csr, ctl, v0, v1, v2, a10, a11, a12, a121, a122):
        self.wc1.set(cvl)
        self.wc2.set(cvr)
        self.wc3.set(csl)
        self.wc4.set(csr)
        self.wc5.set(ctl)
        self.hv1.set(v0  )
        self.hv2.set(v1  )
        self.hv3.set(v2  )
        self.hv4.set(a10 )
        self.hv5.set(a11 )
        self.hv6.set(a12 )
        self.hv7.set(a121)
        self.hv8.set(a122)

        if isinstance(q2, numbers.Number):
            self.kv1.set(q2)
            obs = self.obs.evaluate()
        else:
            obs = []
            for q in q2:
                self.kv1.set(q)
                obs.append(self.obs.evaluate())

        return obs
    
def parameter_cov(ana):
    """
    Get covariance matrix of parameters in EOS analysis object.
    """
    pars = []
    for n in range(0,5):
        rng = np.random.mtrand.RandomState(74205+n)
        p, _ = ana.sample(N=5000, stride=5, pre_N=1000, preruns=5, rng=rng)
        pars += p.tolist()
    pars = np.array(pars)
    cov = np.cov(pars.T).tolist()
    return cov