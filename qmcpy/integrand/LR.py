from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue, Uniform
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError
from numpy import *

class LR(Integrand):

    def __init__(self, sampler, s_matrix, t, prior_mean = 0, prior_variance = 1):
        self.true_measure = Gaussian(sampler, mean=prior_mean, covariance = prior_variance)
        s_matrix2 = array(s_matrix)
        m, d = s_matrix2.shape
        if m != len(t):
            ParameterError("s_matrix must have the same amount of rows as the amount of elements in t")
        c = ones((len(t), 1))
        self.s = column_stack((c, s_matrix))
        if ((t != 0)&(t != 1)).any():
            ParameterError("for all 't_i', t_i can only equal 0 or 1")
        self.t = t
        self.dprime = 2
        super(LR, self).__init__()
        
    def g(self, x, compute_flags):
        st = self.s.transpose()
        z = x@st
        z1 = z*self.t
        matrix = exp(z1)/(1+exp(z))
        matrix1 = prod(matrix, axis=1)
        num = x[:, 0]*matrix1
        n,d=x.shape
        y=zeros((n,2),dtype=float)
        y[:,0] = matrix1
        y[:,1] = num
        return y
