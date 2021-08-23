from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform


class Linear0(Integrand):
    """
    >>> l = Linear0(DigitalNetB2(100,seed=7))
    >>> x = l.discrete_distrib.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean()
    -1.175...e-08
    >>> ytf = l.f(x,periodization_transform='C1SIN')
    >>> ytf.mean()
    -4.050...e-12
    """

    def __init__(self, sampler):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        """
        self.true_measure = Uniform(sampler, lower_bound=-.5, upper_bound=.5)
        self.dprime = 1
        super(Linear0,self).__init__()
    
    def g(self, t):
        y = t.sum(1)
        return y

