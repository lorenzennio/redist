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
        'global_options': { },
        'manual_constraints': constr,
        'priors': [
            { 'parameter': 'B->K^*::alpha^A0_0@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A0_1@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A0_2@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A1_0@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A1_1@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A1_2@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A12_1@BSZ2015', 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^A12_2@BSZ2015', 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^V_0@BSZ2015'  , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^V_1@BSZ2015'  , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^V_2@BSZ2015'  , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T1_0@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T1_1@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T1_2@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T2_1@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T2_2@BSZ2015' , 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T23_0@BSZ2015', 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T23_1@BSZ2015', 'min': -10., 'max': 10., 'type': 'uniform' },
            { 'parameter': 'B->K^*::alpha^T23_2@BSZ2015', 'min': -10., 'max': 10., 'type': 'uniform' },
        ],
        'likelihood': [
            # 'B->K^*::FormFactors[parametric,LCSRLattice]@GKvD:2018A'
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
    # return np.exp(-0.2*q2)
    return 1 - 0.08*np.exp(0.1*q2)

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
        
        self.kv1  = k['q2'                         ]
        self.wc1  = p['sbnunu::Re{cVL}'            ]
        self.wc2  = p['sbnunu::Re{cVR}'            ]
        self.wc3  = p['sbnunu::Re{cSL}'            ]
        self.wc4  = p['sbnunu::Re{cSR}'            ]
        self.wc5  = p['sbnunu::Re{cTL}'            ]
        self.hv01 = p['B->K^*::alpha^V_0@BSZ2015'  ]
        self.hv02 = p['B->K^*::alpha^V_1@BSZ2015'  ]
        self.hv03 = p['B->K^*::alpha^V_2@BSZ2015'  ]
        self.hv04 = p['B->K^*::alpha^A0_0@BSZ2015' ]
        self.hv05 = p['B->K^*::alpha^A0_1@BSZ2015' ]
        self.hv06 = p['B->K^*::alpha^A0_2@BSZ2015' ]
        self.hv07 = p['B->K^*::alpha^A1_0@BSZ2015' ]
        self.hv08 = p['B->K^*::alpha^A1_1@BSZ2015' ]
        self.hv09 = p['B->K^*::alpha^A1_2@BSZ2015' ]
        self.hv10 = p['B->K^*::alpha^A12_1@BSZ2015']
        self.hv11 = p['B->K^*::alpha^A12_2@BSZ2015']
        self.hv12 = p['B->K^*::alpha^T1_0@BSZ2015' ]
        self.hv13 = p['B->K^*::alpha^T1_1@BSZ2015' ]
        self.hv14 = p['B->K^*::alpha^T1_2@BSZ2015' ]
        self.hv15 = p['B->K^*::alpha^T2_1@BSZ2015' ]
        self.hv16 = p['B->K^*::alpha^T2_2@BSZ2015' ]
        self.hv17 = p['B->K^*::alpha^T23_0@BSZ2015']
        self.hv18 = p['B->K^*::alpha^T23_1@BSZ2015']
        self.hv19 = p['B->K^*::alpha^T23_2@BSZ2015']
        
        self.obs = eos.Observable.make('B->K^*nunu::dBR/dq2', p, k, o)

    def distribution(self, q2, cvl, cvr, csl, csr, ctl, v0, v1, v2, a00, a01, a02, a10, a11, a12, a121, a122, t10, t11, t12, t21, t22, t230, t231, t232):
        self.wc1.set(cvl)
        self.wc2.set(cvr)
        self.wc3.set(csl)
        self.wc4.set(csr)
        self.wc5.set(ctl)
        self.hv01.set(v0  )
        self.hv02.set(v1  )
        self.hv03.set(v2  )
        self.hv04.set(a00 )
        self.hv05.set(a01 )
        self.hv06.set(a02 )
        self.hv07.set(a10 )
        self.hv08.set(a11 )
        self.hv09.set(a12 )
        self.hv10.set(a121)
        self.hv11.set(a122)
        self.hv12.set(t10 )
        self.hv13.set(t11 )
        self.hv14.set(t12 )
        self.hv15.set(t21 )
        self.hv16.set(t22 )
        self.hv17.set(t230)
        self.hv18.set(t231)
        self.hv19.set(t232)


        if isinstance(q2, numbers.Number):
            self.kv1.set(q2)
            obs = self.obs.evaluate()
        else:
            obs = []
            for q in q2:
                self.kv1.set(q)
                obs.append(self.obs.evaluate())

        return obs
    
def parameter_cov(ana, chains=5, samples=5000):
    """
    Get covariance matrix of parameters in EOS analysis object.
    """
    pars = []
    for n in range(0,chains):
        rng = np.random.mtrand.RandomState(74205+n)
        p, _ = ana.sample(N=samples, stride=5, pre_N=1000, preruns=5, rng=rng)
        pars += p.tolist()
    pars = np.array(pars)
    cov = np.cov(pars.T).tolist()
    return cov