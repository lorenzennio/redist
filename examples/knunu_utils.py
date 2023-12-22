import numpy as np
import eos
import numbers

def analysis():
    """
    Specify the likelihoods and FF parameter ranges 
    
    Returns:
        EOS analysis instance
    """

    # form factor expansion f+_0,1,2 are expansion parameters up to 2nd order
    # there is no f0_0 because of a constriant which removes one parameter

    parameters = [
        0.33772497529184886, -0.87793473613271, -0.07935870922121949, 
        0.3719622997220613, 0.07388594710238389, 0.327935912834808, 
        -0.9490004115927961, -0.23146429907794228
        ]
    paramerror = [
        0.010131234226468245, 0.09815140228051167, 0.26279803480131697, 
        0.07751034526769873, 0.14588095119443809, 0.019809720318176644, 
        0.16833757660616938, 0.36912754148836896
        ]
    sigma = 15
    analysis_args = {
        'priors': [
            { 'parameter': 'B->K::alpha^f+_0@BSZ2015', 'min': parameters[0]-sigma*paramerror[0], 'max': parameters[0]+sigma*paramerror[0], 'type': 'uniform' },
            { 'parameter': 'B->K::alpha^f+_1@BSZ2015', 'min': parameters[1]-sigma*paramerror[1], 'max': parameters[1]+sigma*paramerror[1], 'type': 'uniform' },
            { 'parameter': 'B->K::alpha^f+_2@BSZ2015', 'min': parameters[2]-sigma*paramerror[2], 'max': parameters[2]+sigma*paramerror[2], 'type': 'uniform' },
            { 'parameter': 'B->K::alpha^f0_1@BSZ2015', 'min': parameters[3]-sigma*paramerror[3], 'max': parameters[3]+sigma*paramerror[3], 'type': 'uniform' },
            { 'parameter': 'B->K::alpha^f0_2@BSZ2015', 'min': parameters[4]-sigma*paramerror[4], 'max': parameters[4]+sigma*paramerror[4], 'type': 'uniform' },
            { 'parameter': 'B->K::alpha^fT_0@BSZ2015', 'min': parameters[5]-sigma*paramerror[5], 'max': parameters[5]+sigma*paramerror[5], 'type': 'uniform' },
            { 'parameter': 'B->K::alpha^fT_1@BSZ2015', 'min': parameters[6]-sigma*paramerror[6], 'max': parameters[6]+sigma*paramerror[6], 'type': 'uniform' },
            { 'parameter': 'B->K::alpha^fT_2@BSZ2015', 'min': parameters[7]-sigma*paramerror[7], 'max': parameters[7]+sigma*paramerror[7], 'type': 'uniform' }
        ],
        'likelihood': [
            'B->K::f_0+f_++f_T@FLAG:2021A',
            'B->K::f_0+f_++f_T@HPQCD:2022A'
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
    return 0.4*np.exp(-0.2*q2)

class null_pred:
    """
    Null (SM) prediction
    """
    def __init__(self):
        p = analysis().parameters
        k = eos.Kinematics({'q2': 0.})
        o = eos.Options(**{'form-factors': 'BSZ2015', 'model': 'WET'})
        
        self.kv1 = k['q2']

        self.obs = eos.Observable.make('B->Knunu::dBR/dq2', p, k, o)

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
        
        self.kv1 = k['q2'                      ]
        self.wc1 = p['sbnunu::Re{cVL}'            ]
        self.wc2 = p['sbnunu::Re{cVR}'            ]
        self.wc3 = p['sbnunu::Re{cSL}'            ]
        self.wc4 = p['sbnunu::Re{cSR}'            ]
        self.wc5 = p['sbnunu::Re{cTL}'            ]
        self.hv1 = p['B->K::alpha^f+_0@BSZ2015']
        self.hv2 = p['B->K::alpha^f+_1@BSZ2015']
        self.hv3 = p['B->K::alpha^f+_2@BSZ2015']
        self.hv4 = p['B->K::alpha^f0_1@BSZ2015']
        self.hv5 = p['B->K::alpha^f0_2@BSZ2015']
        self.hv6 = p['B->K::alpha^fT_0@BSZ2015']
        self.hv7 = p['B->K::alpha^fT_1@BSZ2015']
        self.hv8 = p['B->K::alpha^fT_2@BSZ2015']
        
        self.obs = eos.Observable.make('B->Knunu::dBR/dq2', p, k, o)

    def distribution(self, q2, cvl, cvr, csl, csr, ctl, fp0, fp1, fp2, f01, f02, fT0, fT1, fT2):
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
        self.hv6.set(fT0)
        self.hv7.set(fT1)
        self.hv8.set(fT2)

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