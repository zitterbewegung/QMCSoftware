from .._discrete_distribution import DiscreteDistribution
from .gail_lattice import LatticeGAIL
from .mps_lattice import LatticeMPS
from ...util import ParameterError
from numpy import *


class Lattice(DiscreteDistribution):
    """
    Quasi-Random Lattice nets in base 2.

    >>> l = Lattice(2,seed=7)
    >>> l
    Lattice (DiscreteDistribution Object)
        dimension       2^(1)
        randomize       1
        seed            7
        mimics          StdUniform
        backend         GAIL
    >>> l.gen_samples(4)
    array([[0.076, 0.78 ],
           [0.576, 0.28 ],
           [0.326, 0.53 ],
           [0.826, 0.03 ]])
    >>> l.set_dimension(3)
    >>> l.gen_samples(n_min=4,n_max=8)
    array([[0.563, 0.098, 0.353],
           [0.063, 0.598, 0.853],
           [0.813, 0.848, 0.103],
           [0.313, 0.348, 0.603]])
    >>> Lattice(dimension=2,randomize=False,backend='GAIL').gen_samples(n_min=2,n_max=4)
    array([[0.25, 0.75],
           [0.75, 0.25]])
    >>> Lattice(dimension=2,randomize=False,backend='MPS').gen_samples(n_min=2,n_max=4)
    array([[0.25, 0.75],
           [0.75, 0.25]])

    References:
        
        [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang,
        Lluis Antoni Jimenez Rugama, Da Li, Jagadeeswaran Rathinavel,
        Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3)
        [MATLAB Software], 2019. Available from http://gailgithub.github.io/GAIL_Dev/

        [2] F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients
        - a survey of analysis and implementation, Foundations of Computational Mathematics,
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613

        [3] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/

        [4] Constructing embedded lattice rules for multivariate integration
        R Cools, FY Kuo, D Nuyens -  SIAM J. Sci. Comput., 28(6), 2162-2188.

        [5] L’Ecuyer, Pierre & Munger, David. (2015).
        LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules.
        ACM Transactions on Mathematical Software. 42. 10.1145/2754929.
    """

    parameters = ['dimension','randomize','seed','mimics','backend']

    def __init__(self, dimension=1, randomize=True, seed=None, backend='GAIL', gen_vector_info=None):
        """
        Args:
            dimension (int): dimension of samples
            randomize (bool): If True, apply shift to generated samples. \
                Note: Non-randomized lattice sequence includes the origin.
            seed (int): seed the random number generator for reproducibility
            backend (str): backend generator must be either "GAIL" or "MPS". \
                "GAIL" provides standard point ordering but is slightly slower than "MPS".
            gen_vector_info (dict): if not supplied uses generating vector from [5], \
                otherwise, supply a dictionary with the following keys: \
                    "vector": a numpy.ndarray or list of ints comprising the generating vector. \
                    "n_max": maximum number of samples that can be drawn based on this generating vector. \
                Example: gen_vector_info = {'vector':[1,433461,315689], n_max=2**20}
        """
        self.backend = backend.upper()
        backend_objs = {'GAIL':LatticeGAIL,'MPS':LatticeMPS}
        backend_options = list(backends.keys())
        if self.backend not in backend_options:
            raise ParameterError('Lattice requires backend be in %s'%(str(backend_options)))
        self.generator = backend_objs[self.backend](dimension,generalize,randomize,seed)
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(Lattice, self).__init__()

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate lattice samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples.
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples

        Note:
            Lattice generates in blocks from 2**m to 2**(m+1) so generating
            n_min=3 to n_max=9 requires necessarily produces samples from n_min=2 to n_max=16
            and automatically subsets. May be inefficient for non-powers-of-2 samples sizes.
        """
        if n:
            n_min = 0
            n_max = n
        x = self.generator.gen_samples(n_min,n_max,warn)
        return x

    def set_seed(self, seed):
        """ See abstract method. """
        self.generator.set_seed(seed)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.generator.set_dimension(dimension)
