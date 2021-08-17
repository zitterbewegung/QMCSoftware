from ._true_measure import TrueMeasure
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from numpy import *
from numpy.linalg import cholesky, det, inv, eigh
from scipy.stats import norm
from scipy.special import erfcinv


class Gaussian(TrueMeasure):
    """
    Normal Measure.
    
    >>> g = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2],covariance=[[9,4],[4,5]])
    >>> g.gen_samples(4)
    array([[-0.23979685,  2.98944192],
           [ 2.45994002,  2.17853622],
           [-0.22923897, -1.92667105],
           [ 4.6127697 ,  4.25820377]])
    >>> g
    Gaussian (TrueMeasure Object)
        mean            [1 2]
        covariance      [[9 4]
                        [4 5]]
        decomp_type     PCA
    """

    def __init__(self, sampler, mean=0., covariance=1., decomp_type='PCA'):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            mean (float): mu for Normal(mu,sigma^2)
            covariance (ndarray): sigma^2 for Normal(mu,sigma^2). 
                A float or d (dimension) vector input will be extended to covariance*eye(d)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        self.parameters = ['mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.domain = array([[0,1]])
        self._transform = self._transform_std_uniform
        self._jacobian = self._jacobian_std_uniform
        if isinstance(sampler,DiscreteDistribution) and sampler.mimics=='StdGaussian':
            # need to use transformation from standard gaussian
            self.domain = array([[-inf,inf]])
            self._transform = self._transform_std_gaussian
            self._jacobian = self._jacobian_std_gaussian
        self._parse_sampler(sampler)
        self.decomp_type = decomp_type.upper()
        self._set_mean_cov(mean,covariance)
        self.range = array([[-inf,inf]])
        super(Gaussian,self).__init__()
    
    def _set_mean_cov(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        if isscalar(mean):
            mean = tile(mean,self.d)
        if isscalar(covariance):
            covariance = covariance*eye(self.d)
        self.mu = array(mean)
        self.sigma = array(covariance)
        if self.sigma.shape==(self.d,):
            self.sigma = diag(self.sigma)
        if not (len(self.mu)==self.d and self.sigma.shape==(self.d,self.d)):
            raise DimensionError('''
                    mean must have length d and
                    covariance must be of shape d x d''')
        self._set_constants()
    
    def _set_constants(self):
        if self.decomp_type == 'PCA':
            evals,evecs = eigh(self.sigma) # get eigenvectors and eigenvalues for
            order = argsort(-evals)
            self.a = dot(evecs[:,order],diag(sqrt(evals[order])))
        elif self.decomp_type == 'CHOLESKY':
            self.a = cholesky(self.sigma).T
        else:
            raise ParameterError("decomp_type should be 'PCA' or 'Cholesky'")
        self.det_sigma = det(self.sigma)
        self.det_a = sqrt(self.det_sigma)
        self.inv_sigma = inv(self.sigma)  
    
    def _transform_std_uniform(self, x):
        return self.mu + norm.ppf(x)@self.a.T
    
    def _jacobian_std_uniform(self, x):
        return self.det_a/norm.pdf(norm.ppf(x)).prod(1)
    
    def _transform_std_gaussian(self, x):
        return self.mu + x@self.a.T
    
    def _jacobian_std_gaussian(self, x):
        return tile(self.det_a,x.shape[0])

    def _weight(self, x):
        const = (2*pi)**(-self.d/2) * self.det_sigma**(-1./2)
        delta = x-self.mu
        return const*exp(-((delta@self.inv_sigma)*delta).sum(1)/2)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = Gaussian(sampler,mean=self.mu,covariance=self.covariance,decomp_type=self.decomp_type)
        else:
            m = self.mu[0]
            c = self.sigma[0,0]
            expected_cov = c*eye(int(self.d))
            if not ( (self.mu==m).all() and (self.sigma==expected_cov).all() ):
                raise DimensionError('''
                        In order to spawn a Gaussian measure
                        mean (mu) must be all the same and 
                        covariance must be a scaler times I''')
            spawn = Gaussian(sampler,mean=m,covariance=c,decomp_type=self.decomp_type)
        return spawn
