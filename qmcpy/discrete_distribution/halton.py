from ._discrete_distribution import DiscreteDistribution
from .qrng import ghalton_qrng
from numpy import random


class Halton(DiscreteDistribution):
    """
    Quasi-Random Generalize Halton nets.
    
    References
        Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """

    parameters = ['dimension','generalize','seed']

    def __init__(self, dimension=1, generalize=True, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            generalize (bool): generalize the Halton sequence?
            seed (int): seed the random number generator for reproducibility
        """
        self.dimension = dimension
        self.generalize = generalize
        self.seed = seed
        self.set_seed(self.seed)

    def gen_samples(self, n=8):
        """
        Generate samples

        Args:
            n (int): number of samples

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        x = ghalton_qrng(n,self.dimension,self.generalize,self.seed)
        return x

    def set_seed(self, seed):
        """
        Reseed the generator to get a new scrambling.

        Args:
            seed (int): new seed for generator
        """
        self.seed = seed if seed else random.randint(2**32)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = dimension
    
    def set_generalize(self, generalize):
        """
        Set generalization indicator for Halton sequence
        Args:
            generalize (bool): generalize Halton sequence?
        """
        self.generalize = generalize