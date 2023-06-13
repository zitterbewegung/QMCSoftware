from ..util import ParameterError, MethodImplementationError, _univ_repr, DimensionError
from numpy import *

class DiscreteDistribution(object):
    """ Discrete Distribution abstract class. DO NOT INSTANTIATE. """

    def __init__(self, dimension, seed):
        """
        Args:
            dimension (int or ndarray): dimension of the generator.
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence.
                Note that this is not relevant for IID generators.
            seed (int or numpy.random.SeedSequence): seed to create random number generator
        """
        prefix = 'A concrete implementation of DiscreteDistribution must have '
        if not hasattr(self, 'mimics'):
            raise ParameterError(prefix + 'self.mimcs (measure mimiced by the distribution)')
        if not hasattr(self,'low_discrepancy'):
            raise ParameterError(prefix + 'self.low_discrepancy')
        if not hasattr(self,'parameters'):
            self.parameters = []
        if not hasattr(self,'d_max'):
            raise ParameterError(prefix+ 'self.d_max')
        if isinstance(dimension,list) or isinstance(dimension,ndarray):
            self.dvec = array(dimension)
            self.d = len(self.dvec)
        else:
            self.d = dimension
            self.dvec = arange(self.d)
        if any(self.dvec>self.d_max):
            raise ParameterError('dimension greater than max dimension %d'%self.d_max)
        self._base_seed = seed if isinstance(seed,random.SeedSequence) else random.SeedSequence(seed)
        self.entropy = self._base_seed.entropy
        self.spawn_key = self._base_seed.spawn_key
        self.rng = random.Generator(random.SFC64(self._base_seed))

    def gen_samples(self, *args):
        """
        ABSTRACT METHOD to generate samples from this discrete distribution.

        Args:
            args (tuple): tuple of positional argument. See implementations for details

        Returns:
            ndarray: n x d array of samples
        """
        raise MethodImplementationError(self, 'gen_samples')

    def pdf(self, x):
        """ ABSTRACT METHOD to evaluate pdf of distribution the samples mimic at locations of x. """
        raise MethodImplementationError(self, 'pdf')

    def plot_proj(self, n, d_horizontal = 0, d_vertical = 1, axis=None, math_ind = False, **kwargs):
        """
        Args:
            n(int or array): n is the number of samples that will be plotted or a list of samples(used for extensible point sets)
            d_horizontal (int or array): d_horizontal is a list of dimensions to be plotted on the horizontal axes. Possible input values are from 0 to d-1. Default value is 0 (1st dimension).
            d_vertical (int or array): d_vertical is a list of dimensions to be plotted on the vertical axes for each corresponding element in d_horizontal. Default value is 1 (2nd dimension).
            math_ind : If user wants to pass in the math indices, set it true. By default it is false, so by default this method takes in pyton indices.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import colors
        except:
            raise ImportError("Missing matplotlib.pyplot as plt, Matplotlib must be intalled to run DiscreteDistribution.plot")
        n = atleast_1d(n)
        d_horizontal = atleast_1d(d_horizontal)
        d_vertical = atleast_1d(d_vertical)
        samples = self.gen_samples(n[n.size - 1])
        d = samples.shape[1]
        assert d>=2 

        if axis is None:
            fig, ax = plt.subplots(nrows=d_horizontal.size, ncols=d_vertical.size, figsize=(3*d, 3*d),squeeze=False)                    
            fig.tight_layout(pad=2)
        else:
            ax = axis 
            fig = plt.figure()
            assert (ax.shape[0] >= d_horizontal.size) and (ax.shape[1] >= d_vertical.size)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in range(d_horizontal.size):            
            for j in range(d_vertical.size):
                    n_min = 0
                    for m in range(n.size):
                        n_max = n[m]  
                        if(d_horizontal[i] == d_vertical[j]):
                           ax[i,j].remove()  
                           break
                        if(math_ind is True):
                            x = d_horizontal[i] - 1
                            y = d_vertical[j] - 1
                            x_label = d_horizontal[i]
                            y_label = d_vertical[j]
                        else:
                            x = d_horizontal[i]
                            y = d_vertical[j]
                            x_label = d_horizontal[i] + 1
                            y_label = d_vertical[j] + 1
                        ax[i,j].scatter(samples[n_min:n_max,x],samples[n_min:n_max,y],s=5,color=colors[m],label='n_min = %d, n_max = %d'%(n_min,n_max),**kwargs)          
                        ax[i,j].set_aspect(1)
                        ax[i,j].set_xlabel(r'$x_{i%d}$'%(x_label)); ax[i,j].set_ylabel(r'$x_{i%d}$'%(y_label))
                        ax[i,j].set_xlim([0,1]); ax[i,j].set_ylim([0,1])
                        ax[i,j].set_xticks([0,1]); ax[i,j].set_yticks([0,1]) 
                        n_min = n[m]
        return fig, ax

    def spawn(self, s=1, dimensions=None):
        """
        Spawn new instances of the current discrete distribution but with new seeds and dimensions.
        Developed for multi-level and multi-replication (Q)MC algorithms.

        Args:
            s (int): number of spawn
            dimensions (ndarray): length s array of dimension for each spawn. Defaults to current dimension

        Return:
            list: list of DiscreteDistribution instances with new seeds and dimensions
        """
        if (isinstance(dimensions,list) or isinstance(dimensions,ndarray)) and len(dimensions)==s:
            dimensions = array(dimensions)
        elif isscalar(dimensions) and dimensions%1==0:
            dimensions = tile(dimensions,s)
        elif dimensions is None:
            dimensions = tile(self.d,s)
        else:
            raise ParameterError("invalid spawn dimensions, must be None, int, or length s ndarray")
        child_seeds = self._base_seed.spawn(s)
        return [self._spawn(child_seeds[i],dimensions[i]) for i in range(s)]

    def _spawn(self, child_seed, dimension):
        """
        ABSTRACT METHOD, used by self.spawn

        Args:
            child_seeds (numpy.random.SeedSequence): length s array of seeds for each spawn
            dimension (int): lenth s array of dimensions for each spawn

        Return:
            DiscreteDistribution: spawn with new dimension using child_seed
        """
        raise MethodImplementationError(self, '_spawn')

    def __repr__(self):
        return _univ_repr(self, "DiscreteDistribution", ['d']+self.parameters+['entropy','spawn_key'])

    def __call__(self, *args, **kwargs):
        if len(args)>2 or len(args)==0:
            raise Exception('''
                expecting 1 or 2 arguments:
                    1 argument corresponds to n, the number of smaples to generate. In this case n_min=0 and n_max=n for LD sequences
                    2 arguments corresponds to n_min and n_max. Note this is incompatible with IID generators which only expect 1 argument.
                ''')
        if len(args) == 1:
            return self.gen_samples(n=args[0])
        else:
            return self.gen_samples(n_min=args[0],n_max=args[1])

class LD(DiscreteDistribution): pass

class IID(DiscreteDistribution): pass