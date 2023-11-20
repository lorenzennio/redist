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

    analysis_args = {
        'priors': [
            { 'parameter': 'B->D::alpha^f+_0@BSZ2015', 'min':  0.0, 'max':  1.0, 'type': 'uniform' },
            { 'parameter': 'B->D::alpha^f+_1@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->D::alpha^f+_2@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->D::alpha^f0_1@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' },
            { 'parameter': 'B->D::alpha^f0_2@BSZ2015', 'min': -5.0, 'max': +5.0, 'type': 'uniform' }
        ],
        'likelihood': [
            'B->D::f_++f_0@HPQCD:2015A',
            'B->D::f_++f_0@FNAL+MILC:2015B'
        ]
    }

    analysis = eos.Analysis(**analysis_args)
    analysis.optimize()
    return analysis

class null_pred:
    """
    Null (SM) prediction
    """
    def __init__(self):
        p = eos.Parameters()
        o = eos.Options({'form-factors': 'BSZ2015', 'l': 'tau', 'model':'WET'})
        k = eos.Kinematics({
            'q2':            5.0,  'q2_min':            3.4,     'q2_max':           26.41,
            'cos(theta_l)':  0.0,  'cos(theta_l)_min': -1.0,      'cos(theta_l)_max': +1.0,
        })
        self.kv1 = k['q2']
        self.kv2 = k['cos(theta_l)']
        self.pdf = eos.SignalPDF.make('B->pilnu::d^2Gamma/dq2/dcos(theta_l)', p, k, o)


    def distribution(self, q2, costl):
        if isinstance(q2, numbers.Number) and isinstance(costl, numbers.Number):
            self.kv1.set(q2)
            self.kv2.set(costl)
            obs = np.exp(self.pdf.evaluate() - self.pdf.normalization())
        else:
            obs = []
            for q in q2:
                coslist = []
                for ct in costl:
                    self.kv1.set(q)
                    self.kv2.set(ct)
                    o = np.exp(self.pdf.evaluate() - self.pdf.normalization())
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
        k = eos.Kinematics({
            'q2':            5.0,  'q2_min':            3.4,     'q2_max':           26.41,
            'cos(theta_l)':  0.0,  'cos(theta_l)_min': -1.0,      'cos(theta_l)_max': +1.0,
        })
        self.kv1 = k['q2']
        self.kv2 = k['cos(theta_l)']
        self.wc1 = p['ubtaunutau::Re{cVL}'     ]
        self.wc2 = p['ubtaunutau::Re{cSL}'     ]
        self.wc3 = p['ubtaunutau::Re{cT}'      ]
        self.hv1 = p['B->D::alpha^f+_0@BSZ2015']
        self.hv2 = p['B->D::alpha^f+_1@BSZ2015']
        self.hv3 = p['B->D::alpha^f+_2@BSZ2015']
        self.hv4 = p['B->D::alpha^f0_1@BSZ2015']
        self.hv5 = p['B->D::alpha^f0_2@BSZ2015']
        self.pdf = eos.SignalPDF.make('B->pilnu::d^2Gamma/dq2/dcos(theta_l)', p, k, o)


    def distribution(self, q2, costl, cvl, csl, ct, fp0, fp1, fp2, f01, f02):
        self.wc1.set(cvl)
        self.wc2.set(csl)
        self.wc3.set(ct )
        self.hv1.set(fp0)
        self.hv2.set(fp1)
        self.hv3.set(fp2)
        self.hv4.set(f01)
        self.hv5.set(f02)

        if isinstance(q2, numbers.Number) and isinstance(costl, numbers.Number):
            self.kv1.set(q2)
            self.kv2.set(costl)
            obs = np.exp(self.pdf.evaluate() - self.pdf.normalization())
        else:
            obs = []
            for q in q2:
                coslist = []
                for ct in costl:
                    self.kv1.set(q)
                    self.kv2.set(ct)
                    o = np.exp(self.pdf.evaluate() - self.pdf.normalization())
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