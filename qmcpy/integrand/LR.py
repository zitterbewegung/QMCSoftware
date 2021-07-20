from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue, Uniform
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError
from numpy import *

class LR(Integrand):

    def __init__(self, sampler, s_matrix, t, prior_mean, prior_variance):
        self.true_measure = Gaussian(sampler, mean=prior_mean, covariance = prior_variance)
        s_matrix2 = array(s_matrix)
        m, d = s_matrix2.shape
        check1 = True
        if m != len(t):
            check1 = False
        if check1 == True:
            c = ones((len(t), 1))
            self.s = column_stack((c, s_matrix))
        else:
            print("s_matrix must have the same amount of rows as the amount of elements in t")
        check = True
        for i in range(len(t)):
            if 0 != t[i] and t[i] != 1:
                check = False
        if check == True:
            self.t = t
        else:
            print("for all 't_i', t_i can only equal 0 or 1")
        self.dprime = 1
        super(LR, self).__init__()
        
    def g(self, x):
        st = self.s.transpose()
        z = x@st
        z1 = z*self.t
        matrix = exp(z1)/(1+exp(z))
        matrix1 = prod(matrix, axis=1)
        return matrix1