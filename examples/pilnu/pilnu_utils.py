import numpy as np
import eos
import numbers

def analysis():
    """
    Specify the likelihoods and FF parameter ranges 
    
    Returns:
        EOS analysis instance
    """

    analysis_args = {
        'priors': [      
            { 'parameter': 'B->pi::alpha^f+_0@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->pi::alpha^f+_1@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->pi::alpha^f+_2@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->pi::alpha^f0_1@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->pi::alpha^f0_2@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->pi::alpha^fT_0@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->pi::alpha^fT_1@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->pi::alpha^fT_2@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
        ],
        'likelihood': [
            'B->pi::FormFactors[parametric,LCSRLattice]@GKvD:2018A'
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
    return 1 - 0.08*np.exp(0.1*q2)

class null_pred:
    """
    Null (SM) prediction
    """
    def __init__(self):
        self.ana = analysis()
        p = self.ana.parameters
        o = eos.Options({'form-factors': 'BSZ2015', 'l': 'tau', 'model':'WET'})
        k = eos.Kinematics({'q2': 5.0, 'cos(theta_l)': 0.0,})

        self.kv1 = k['q2']
        self.kv2 = k['cos(theta_l)']
        
        self.obs = eos.Observable.make('B->pilnu::d^2BR/dq2/dcos(theta_l)', p, k, o)


    def distribution(self, q2, costl):
        if isinstance(q2, numbers.Number) and isinstance(costl, numbers.Number):
            self.kv1.set(q2)
            self.kv2.set(costl)
            obs = self.obs.evaluate()
        else:
            obs = []
            for q in q2:
                coslist = []
                for ct in costl:
                    self.kv1.set(q)
                    self.kv2.set(ct)
                    o = self.obs.evaluate()
                    coslist.append(o)
                obs.append(coslist)
            obs = np.array(obs).T

        return obs
    
class alt_pred:
    """
    Alternative (BSM) prediction
    """
    def __init__(self):
        self.ana = analysis()
        p = self.ana.parameters
        o = eos.Options({'form-factors': 'BSZ2015', 'l': 'tau', 'model':'WET'})
        k = eos.Kinematics({'q2': 5.0, 'cos(theta_l)': 0.0,})
        
        self.kv1 = k['q2']
        self.kv2 = k['cos(theta_l)']
        self.wc1 = p['ubtaunutau::Re{cVL}']
        self.wc2 = p['ubtaunutau::Re{cVR}']
        self.wc3 = p['ubtaunutau::Re{cSL}']
        self.wc4 = p['ubtaunutau::Re{cSR}']
        self.wc5 = p['ubtaunutau::Re{cT}' ]
        self.hv1 = p['B->pi::alpha^f+_0@BSZ2015']
        self.hv2 = p['B->pi::alpha^f+_1@BSZ2015']
        self.hv3 = p['B->pi::alpha^f+_2@BSZ2015']
        self.hv4 = p['B->pi::alpha^f0_1@BSZ2015']
        self.hv5 = p['B->pi::alpha^f0_2@BSZ2015']
        self.hv6 = p['B->pi::alpha^fT_0@BSZ2015']
        self.hv7 = p['B->pi::alpha^fT_1@BSZ2015']
        self.hv8 = p['B->pi::alpha^fT_2@BSZ2015']
        
        self.obs = eos.Observable.make('B->pilnu::d^2BR/dq2/dcos(theta_l)', p, k, o)


    def distribution(self, q2, costl, cvl, cvr, csl, csr, ctl, fp0, fp1, fp2, f01, f02, ft0, ft1, ft2):
        self.wc1.set(cvl)
        self.wc2.set(cvr)
        self.wc3.set(csl)
        self.wc4.set(csr)
        self.wc5.set(ctl)
        self.hv1.set(fp0)
        self.hv2.set(fp1)
        self.hv3.set(fp2)
        self.hv4.set(f01)
        self.hv5.set(f02)
        self.hv6.set(ft0)
        self.hv7.set(ft1)
        self.hv8.set(ft2)

        if isinstance(q2, numbers.Number) and isinstance(costl, numbers.Number):
            self.kv1.set(q2)
            self.kv2.set(costl)
            obs = self.obs.evaluate()
        else:
            obs = []
            for q in q2:
                coslist = []
                for ct in costl:
                    self.kv1.set(q)
                    self.kv2.set(ct)
                    o = self.obs.evaluate()
                    coslist.append(o)
                obs.append(coslist)
            obs = np.array(obs).T

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