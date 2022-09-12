from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarDataRep
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import Lattice,DigitalNetB2,Halton
from ..discrete_distribution._discrete_distribution import LD
from ..true_measure import Gaussian,Uniform
from ..integrand import Keister,BoxIntegral,CustomFun
from ..util import MaxSamplesWarning, NotYetImplemented, ParameterWarning, ParameterError
from numpy import *
from scipy.stats import t
from time import time
import warnings


class CubQMCCLT(StoppingCriterion):
    """
    Stopping criterion based on Central Limit Theorem for multiple replications.
    
    >>> k = Keister(Lattice(seed=7))
    >>> sc = CubQMCCLT(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    array([1.38030146])
    >>> data
    MeanVarDataRep (AccumulateData Object)
        solution        1.380
        indv_error      7.92e-04
        ci_low          1.380
        ci_high         1.381
        ci_comb_low     1.380
        ci_comb_high    1.381
        flags_comb      1
        flags_indv      1
        n_total         2^(12)
        n               2^(12)
        n_rep           2^(8)
        time_integrate  ...
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
        replications    2^(4)
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    Lattice (DiscreteDistribution Object)
        d               1
        dvec            0
        randomize       1
        order           natural
        entropy         7
        spawn_key       ()
    >>> f = BoxIntegral(Lattice(3,seed=7), s=[-1,1])
    >>> abs_tol = 1e-3
    >>> sc = CubQMCCLT(f, abs_tol=abs_tol)
    >>> solution,data = sc.integrate()
    >>> solution
    array([1.19023153, 0.96068581])
    >>> data
    MeanVarDataRep (AccumulateData Object)
        solution        [1.19  0.961]
        indv_error      [0.001 0.001]
        ci_low          [1.19 0.96]
        ci_high         [1.191 0.961]
        ci_comb_low     [1.19 0.96]
        ci_comb_high    [1.191 0.961]
        flags_comb      [ True  True]
        flags_indv      [ True  True]
        n_total         2^(21)
        n               [2097152.    8192.]
        n_rep           [131072.    512.]
        time_integrate  ...
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.001
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
        replications    2^(4)
    BoxIntegral (Integrand Object)
        s               [-1  1]
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    Lattice (DiscreteDistribution Object)
        d               3
        dvec            [0 1 2]
        randomize       1
        order           natural
        entropy         7
        spawn_key       ()
    >>> sol3neg1 = -pi/4-1/2*log(2)+log(5+3*sqrt(3))
    >>> sol31 = sqrt(3)/4+1/2*log(2+sqrt(3))-pi/24
    >>> true_value = array([sol3neg1,sol31])
    >>> (abs(true_value-solution)<abs_tol).all()
    True
    >>> cf = CustomFun(
    ...     true_measure = Uniform(DigitalNetB2(6,seed=7)),
    ...     g = lambda x,compute_flags=None: (2*arange(1,7)*x).reshape(-1,2,3),
    ...     rho = (2,3))
    >>> sol,data = CubQMCCLT(cf,abs_tol=1e-4).integrate()
    >>> data
    MeanVarDataRep (AccumulateData Object)
        solution        [[1. 2. 3.]
                        [4. 5. 6.]]
        indv_error      [[2.842e-05 0.000e+00 0.000e+00]
                        [6.530e-06 2.491e-10 0.000e+00]]
        ci_low          [[1. 2. 3.]
                        [4. 5. 6.]]
        ci_high         [[1. 2. 3.]
                        [4. 5. 6.]]
        ci_comb_low     [[1. 2. 3.]
                        [4. 5. 6.]]
        ci_comb_high    [[1. 2. 3.]
                        [4. 5. 6.]]
        flags_comb      [[ True  True  True]
                        [ True  True  True]]
        flags_indv      [[ True  True  True]
                        [ True  True  True]]
        n_total         2^(14)
        n               [[ 4096.  4096.  4096.]
                        [16384.  4096.  4096.]]
        n_rep           [[ 256.  256.  256.]
                        [1024.  256.  256.]]
        time_integrate  ...
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         1.00e-04
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
        replications    2^(4)
    CustomFun (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    DigitalNetB2 (DiscreteDistribution Object)
        d               6
        dvec            [0 1 2 3 4 5]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=256., n_max=2**30,
        inflate=1.2, alpha=0.01, replications=16., 
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol)):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (ndarray): significance level for confidence interval
            abs_tol (ndarray): absolute error tolerance
            rel_tol (ndarray): relative error tolerance
            n_max (int): maximum number of samples
            replications (int): number of replications
            error_fun: function taking in the approximate solution vector, 
                absolute tolerance, and relative tolerance which returns the approximate error. 
                Default indicates integration until either absolute OR relative tolerance is satisfied.
        """
        self.parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max','replications']
        # Input Checks
        if log2(n_init) % 1 != 0:
            warning_s = ' n_init must be a power of 2. Using n_init = 32'
            warnings.warn(warning_s, ParameterWarning)
            n_init = 32
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.alpha = alpha
        self.inflate = float(inflate)
        self.replications = int(replications)
        self.error_fun = error_fun
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.rho = self.integrand.rho
        self.d = self.discrete_distrib.d
        super(CubQMCCLT,self).__init__(allowed_levels=["single"], allowed_distribs=[LD], allow_vectorized_integrals=True)
        if not self.discrete_distrib.randomize:
            raise ParameterError("CLTRep requires distribution to have randomize=True")
        self.alphas_indv,identity_dependency = self._compute_indv_alphas(full(self.integrand.eta,self.alpha))
        self.t_star = -t.ppf(self.alphas_indv/2,df=self.replications-1)
         
    def integrate(self):
        """ See abstract method. """
        t_start = time()
        self.datum = empty(self.rho,dtype=object)
        for j in ndindex(self.rho):
            self.datum[j] = MeanVarDataRep(self.t_star[j],self.inflate,self.replications)
        self.data = MeanVarDataRep.__new__(MeanVarDataRep)
        self.data.flags_indv = tile(False,self.rho)
        self.data.compute_flags = tile(True,self.rho)
        self.data.rep_distribs = self.integrand.discrete_distrib.spawn(s=self.replications)
        self.data.n_rep = tile(self.n_init,self.rho)
        self.data.n_min_rep = 0
        self.data.ci_low = tile(-inf,self.rho)
        self.data.ci_high = tile(inf,self.rho)
        self.data.solution_indv = tile(nan,self.rho)
        self.data.solution = nan
        self.data.xfull = empty((0,self.d))
        self.data.yfull = empty((0,)+self.rho)
        while True:
            n_min = self.data.n_min_rep
            n_max = int(self.data.n_rep.max())
            n = int(n_max-n_min)
            xnext = vstack([self.data.rep_distribs[r].gen_samples(n_min=n_min,n_max=n_max) for r in range(self.replications)])
            ynext = self.integrand.f(xnext,compute_flags=self.data.compute_flags)
            for j in ndindex(self.rho):
                if self.data.flags_indv[j]: continue
                yj = ynext[(slice(None),)+j].reshape((n,self.replications),order='f')
                self.data.solution_indv[j],self.data.ci_low[j],self.data.ci_high[j] = self.datum[j].update_data(yj)
            self.data.xfull = vstack((self.data.xfull,xnext))
            self.data.yfull = vstack((self.data.yfull,ynext))
            self.data.indv_error = (self.data.ci_high-self.data.ci_low)/2
            self.data.ci_comb_low,self.data.ci_comb_high = self.integrand.bound_fun(self.data.ci_low,self.data.ci_high)
            self.abs_tols,self.rel_tols = full_like(self.data.ci_comb_low,self.abs_tol),full_like(self.data.ci_comb_low,self.rel_tol)
            fidxs = isfinite(self.data.ci_comb_low)&isfinite(self.data.ci_comb_high)
            slow,shigh,abs_tols,rel_tols = self.data.ci_comb_low[fidxs],self.data.ci_comb_high[fidxs],self.abs_tols[fidxs],self.rel_tols[fidxs]
            self.data.solution = tile(nan,self.data.ci_comb_low.shape)
            self.data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            self.data.flags_comb = tile(False,self.data.ci_comb_low.shape)
            self.data.flags_comb[fidxs] = (shigh-slow) <= (self.error_fun(slow,abs_tols,rel_tols)+self.error_fun(shigh,abs_tols,rel_tols))
            self.data.flags_indv = self.integrand.dependency(self.data.flags_comb)
            self.data.compute_flags = ~self.data.flags_indv
            self.data.n = self.replications*self.data.n_rep
            self.data.n_total = self.data.n.max()
            if sum(self.data.compute_flags)==0:
                break # sufficiently estimated
            elif 2*self.data.n_total>self.n_max:
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples would exceeds n_max = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied. """ \
                % (int(self.data.n_total),int(self.data.n_total),int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                self.data.n_min_rep = n_max
                self.data.n_rep += self.data.n_rep*(self.data.compute_flags) # double sample size
        self.data.integrand = self.integrand
        self.data.true_measure = self.true_measure
        self.data.discrete_distrib = self.discrete_distrib
        self.data.stopping_crit = self
        self.data.parameters = [
            'solution',
            'indv_error',
            'ci_low',
            'ci_high',
            'ci_comb_low',
            'ci_comb_high',
            'flags_comb',
            'flags_indv',
            'n_total',
            'n',
            'n_rep',
            'time_integrate']
        self.data.datum = self.datum
        self.data.time_integrate = time()-t_start
        return self.data.solution,self.data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol != None: self.abs_tol = abs_tol
        if rel_tol != None: self.rel_tol = rel_tol

class CubQMCRep(CubQMCCLT): pass
