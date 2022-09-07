from .gaussian import Gaussian
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import DigitalNetB2
from ._true_measure import TrueMeasure
from ..util import ParameterError, _univ_repr
from numpy import *


class BrownianMotion(Gaussian):
    """
    Geometric Brownian Motion.
    
    >>> bm = BrownianMotion(DigitalNetB2(4,seed=7),t_final=2,drift=2)
    >>> bm.gen_samples(2)
    array([[1.223807  , 2.38808042, 2.67953128, 3.46144448],
           [0.41066557, 2.38751411, 4.02608053, 4.27243528]])
    >>> bm
    BrownianMotion (TrueMeasure Object)
        time_vec        [0.5 1.  1.5 2. ]
        drift           2^(1)
        mean            [1. 2. 3. 4.]
        covariance      [[0.5 0.5 0.5 0.5]
                        [0.5 1.  1.  1. ]
                        [0.5 1.  1.5 1.5]
                        [0.5 1.  1.5 2. ]]
        decomp_type     PCA
    """

    def __init__(self, sampler, t_final=1, drift=0, decomp_type='PCA'):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            t_final (float): end time for the Brownian Motion. 
            drift (int): Gaussian mean is time_vec*drift
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        self.parameters = ['time_vec', 'drift', 'mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.domain = array([[0,1]])
        self._parse_sampler(sampler)
        self.t = t_final # exercise time
        self.drift = drift
        self.time_vec = linspace(self.t/self.d,self.t,self.d) # evenly spaced
        self.sigma_bm = array([[min(self.time_vec[i],self.time_vec[j]) for i in range(self.d)] for j in range(self.d)])
        self.drift_time_vec = self.drift*self.time_vec # mean
        self._parse_gaussian_params(self.drift_time_vec,self.sigma_bm,decomp_type)

        self.range = array([[-inf,inf]])
        super(Gaussian,self).__init__()
    
    def _spawn(self, sampler, dimension):
        return BrownianMotion(sampler,t_final=self.t,drift=self.drift,decomp_type=self.decomp_type)
        
