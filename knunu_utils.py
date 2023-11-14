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
    return np.exp(-0.2*q2)

class null_pred:
    """
    Null (SM) prediction
    """
    def __init__(self):
        self.parameters = eos.Parameters()
        self.options = eos.Options(**{'form-factors': 'BSZ2015', 'model': 'WET'})

    def distribution(self, q2):
        if isinstance(q2, numbers.Number):
            obs = eos.Observable.make(
                'B->Knunu::dBR/dq2', 
                self.parameters, 
                eos.Kinematics(q2=q2),
                self.options).evaluate()
        else:
            obs = np.array([eos.Observable.make(
                'B->Knunu::dBR/dq2', 
                self.parameters, 
                eos.Kinematics(q2=q),
                self.options).evaluate() 
                   for q in q2])
            
        return obs
    
class alt_pred:
    """
    Alternative (BSM) prediction
    """
    def __init__(self):
        self.ana = analysis()
        self.options = eos.Options(**{'form-factors': 'BSZ2015', 'model': 'WET'})

    def distribution(self, q2, cvl, csl, ctl, fp0, fp1, fp2, f01, f02, fT0, fT1, fT2):
        self.ana.parameters['sbnunu::Re{cVL}'         ].set(cvl)
        self.ana.parameters['sbnunu::Re{cSL}'         ].set(csl)
        self.ana.parameters['sbnunu::Re{cTL}'         ].set(ctl)
        self.ana.parameters['B->K::alpha^f+_0@BSZ2015'].set(fp0)
        self.ana.parameters['B->K::alpha^f+_1@BSZ2015'].set(fp1)
        self.ana.parameters['B->K::alpha^f+_2@BSZ2015'].set(fp2)
        self.ana.parameters['B->K::alpha^f0_1@BSZ2015'].set(f01)
        self.ana.parameters['B->K::alpha^f0_2@BSZ2015'].set(f02)
        self.ana.parameters['B->K::alpha^fT_0@BSZ2015'].set(fT0)
        self.ana.parameters['B->K::alpha^fT_1@BSZ2015'].set(fT1)
        self.ana.parameters['B->K::alpha^fT_2@BSZ2015'].set(fT2)

        if isinstance(q2, numbers.Number):
            obs = eos.Observable.make(
                'B->Knunu::dBR/dq2', 
                self.ana.parameters, 
                eos.Kinematics(q2=q2),
                self.options).evaluate()
        else:
            obs = np.array([eos.Observable.make(
                'B->Knunu::dBR/dq2', 
                self.ana.parameters, 
                eos.Kinematics(q2=q),
                self.options).evaluate() 
                   for q in q2])
            
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