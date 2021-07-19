from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue, Uniform
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError
from numpy import *

class LR(Integrand):

    """
    $f(\\boldsymbol{t}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{t} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1./2.

    >>> s = numpy.array([[0]])
    >>> t = numpy.array([1])
    >>> no, dim = s.shape
    >>> my_instance = LR(sampler = Sobol(dimension=dim+1, seed = 7), s_matrix = s, t = t)
    >>> p = my_instance.discrete_distrib.gen_samples(n_min=0, n_max=2**10)
    >>> y = my_instance.f(p)
    >>> print(y.mean())
    0.6201145005168884
    >>> my_instance.true_measure
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    (qmcpy) 
    """
    def __init__(self, sampler, s_matrix, t):
        self.true_measure = Uniform(sampler)
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
            t1 = zeros((len(t), len(t)))
            for i in range(len(t)):
                t1[i][i] = t[i]
            self.t = t1
        else:
            print("for all 't_i', t_i can only equal 0 or 1")
        self.dprime = 1
        super(LR, self).__init__()
        
    def g(self, x):
        st = self.s.transpose()
        z = x@st
        z1 = z@self.t
        matrix = exp(z1)/(1+exp(z))
        matrix1 = prod(matrix, axis=1)
        return matrix1