from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError, ParameterWarning
from ..c_lib import c_lib
import ctypes
from os.path import dirname, abspath, isfile
from numpy import *
import warnings


class DigitalNetB2(DiscreteDistribution):
    """
    Quasi-Random digital nets in base 2.
    
    >>> s = DigitalNetB2(2,seed=7)
    >>> s.gen_samples(4)
    array([[0.24423122, 0.31666099],
           [0.55776053, 0.95317207],
           [0.26378847, 0.56871143],
           [0.9523192 , 0.20515005]])
    >>> s.gen_samples(1)
    array([[0.24423122, 0.31666099]])
    >>> s
    DigitalNetB2 (DiscreteDistribution Object)
        d               2^(1)
        randomize       LMS_DS
        graycode        0
        seed            7
        mimics          StdUniform
        dvec            [0 1]
    >>> DigitalNetB2(dimension=2,randomize=False,graycode=True).gen_samples(n_min=2,n_max=4)
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> DigitalNetB2(dimension=2,randomize=False,graycode=False).gen_samples(n_min=2,n_max=4)
    array([[0.25, 0.75],
           [0.75, 0.25]])
           
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

        [2] Faure, Henri, and Christiane Lemieux. 
        “Implementation of Irreducible Sobol' Sequences in Prime Power Bases.” 
        Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.

        [3] F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
        - a survey of analysis and implementation, Foundations of Computational Mathematics, 
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613
        
        [4] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/

        [5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. 
        (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
        In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), 
        Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. 
        Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

        [6] I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: 
        "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, 
        Russian Acamdey of Sciences, Moscow (1992).

        [7] Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 

        [8] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. 
        DOI:https://doi.org/10.1145/42288.214372
    """

    dnb2_cf = c_lib.gen_digitalnetb2
    dnb2_cf.argtypes = [
        ctypes.c_ulong,  # n
        ctypes.c_ulong, # n0
        ctypes.c_uint32,  # d
        ctypes.c_uint32, # graycode
        ctypes.c_uint32, # m_max
        ctypes.c_uint32, # t_max
        ctypeslib.ndpointer(ctypes.c_uint64, flags='C_CONTIGUOUS'),  # znew
        ctypes.c_uint32, # set_rshift
        ctypeslib.ndpointer(ctypes.c_uint64, flags='C_CONTIGUOUS'),  # rshift
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # x
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]  # xr
    dnb2_cf.restype = ctypes.c_uint32

    def __init__(self, dimension=1, randomize='LMS_DS', graycode=False, seed=None, z_path='sobol_mat.21201.32.32.msb.npy', t_lms=None, verbose=False):
        """
        Args:
            dimension (int): dimension of samples
            randomize (bool): apply randomization? True defaults to LMS_DS. Can also explicitly pass in
                'LMS_DS': linear matrix scramble with digital shift
                'LMS': linear matrix scramble only
                'DS': digital shift only
            graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
            seed (list): int seed of list of seeds, one for each dimension.
            z_path (str): path to generating matrices. 
                z_path sould be formatted like `gen_mat.21201.32.msb.npy` with name.d_max.m_max.msb_or_lsb.npy
            t_lms (int): LMS scrambling matrix will be t_max x t for generating matrix of shape t x m
            verbose (bool): print randomization details
        """
        self.parameters = ['d','randomize','graycode','seed','mimics','dvec']
        # set generating matrix
        z_root = dirname(abspath(__file__))+'/generating_matrices/'
        if isfile(z_path):
            self.z = load(z_path).astype(uint64)
        elif isfile(z_root+z_path):
            self.z = load(z_root+z_path).astype(uint64)
        else:
            raise ParameterError('z_path `' + z_path + '` not found. ')
        f = z_path.split('/')[-1]
        f_lst = f.split('.')
        self.d_max = int(f_lst[1])
        self.m_max = int(f_lst[2])
        self.t_max = int(f_lst[3])
        msblsb = f_lst[4].lower()
        if msblsb == 'msb':
            self.msb = 1
        elif msblsb == 'lsb':
            self.msb = 0
        else:
            msg = '''
                z_path sould be formatted like `name.d_max.m_max.t_max.msb_or_lsb.npy`
                    d_max is the max dimension, 
                    m_max is such that 2^m_max is the max number of samples supported 
                    t_max is the number of bits in each int of the generating matrix
                    msb_or_lsb is how each int encodes the binary vector
                        e.g. 6 is [1 1 0] in msb and [0 1 1] in lsb
            '''
            raise ParameterError(msg)
        # set parameters
        if randomize==None or (isinstance(randomize,str) and (randomize.upper()=='NONE' or randomize.upper=='NO')):
            self.set_lms = False
            self.set_rshift = False
        elif isinstance(randomize,bool):
            if randomize:
                self.set_lms = True
                self.set_rshift = True
            else:
                self.set_lms = False
                self.set_rshift = False
        elif randomize.upper() == 'LMS_DS':
            self.set_lms = True
            self.set_rshift = True
        elif randomize.upper() == 'LMS':
            self.set_lms = True
            self.set_rshift = False
        elif randomize.upper() == "DS":
            self.set_lms = False
            self.set_rshift = True
        else:
            msg = '''
                DigitalNetB2' randomize should be either 
                    "LMS_DS" for linear matrix scramble with digital shift or
                    'LMS' for linear matrix scramble only or
                    'DS' for digital shift only. 
            '''
            raise ParameterError(msg)
        self.t2_max = t_lms  if t_lms and self.set_lms else self.t_max
        self.randomize = randomize
        self.graycode = graycode
        if isinstance(dimension,list) or isinstance(dimension,ndarray):
            self.dvec = array(dimension)
            self.d = len(self.dvec)
        else:
            self.d = dimension
            self.dvec = arange(self.d)
        self.verbose = verbose
        self.set_seed(seed) # calls self.set_seed(self.seed)
        self.errors = {
            1: 'using natural ordering (graycode=0) where n0 and/or (n0+n) is not 0 or a power of 2 is not allowed.',
            2: 'Exceeding max samples (2^%d) or max dimensions (%d).'%(self.m_max,self.d_max)}
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(DigitalNetB2,self).__init__()        

    def set_digitalnetb2_randomizations(self, dvec, d, set_lms, set_rshift, seeds, 
        d_max, m_max, t_max, t2_max, z, msb, print_scramble_mat=False):
        """
        initialize digital net generator in base 2 by
            - flipping least significant bit (lsb) order to most significant bit (msb) order
            - optionally applying a linear matrix scramble (lms) to the generating vector
            - optionally creating a vector of random (digital) shifts 
        Args:
            dvec (ndarray uint32): length d vector of dimesions
            d (int): length(dvec) = number of dimensions
            set_lms (bool): lms flag
            set_rshift (bool): random shift flag
            seeds (ndarray uint32): length d vector of seeds, one for each dimension in dvec
            d_max (uint32): max supported dimension 
            m_max (uint32): 2^m_max is the maximum number of samples supported
            t_max (uint32): number of bits in each element of z, the generating vector
            t2_max (uint32): number of rows in the lms matrix
            z (ndarray uint64): original generating vector with shape d_max x m_max with each int having t_max bits
            msb (bool): msb flag  e.g. 6 encodes [1 1 0] in msb and [0 1 1] in lsb
            print_scramble_mat (bool): flag to print the scrambling matrix
        
        Return:
            znew (ndarray uint64): d x m_max generating vector in msb order, possibly with lms applied, for gen_digitalnetb2
            rshift (ndarray uint64): length d vector of random digital shifts for gen_digitalnetb2
        
        See: https://bitbucket.org/dnuyens/qmc-generators/src/cb0f2fb10fa9c9f2665e41419097781b611daa1e/cpp/digitalseq_b2g.hpp
        """
        # parameter checks
        if (dvec>d_max).any():
            raise Exception("require (dvec <= d_max).all()")
        if t_max>64 or t2_max>64 or t_max>t2_max:
            raise Exception("require t_max <= t2_max <= 64")
        # constants
        randomizer = random.RandomState()
        znew = zeros((d,m_max),dtype=uint64)
        rshift = zeros(d,dtype=uint64)
        zmsb = z[dvec,:]
        # flip bits if using lsb (least significant bit first) order
        if not msb:
            for j in range(d):
                for m in range(m_max):
                    zmsb[j,m] = self._flip_bits(zmsb[j,m],t_max)
        # set the linear matrix scrambling and random shift
        if set_lms and print_scramble_mat: print('s (scrambling_matrix)')
        for j in range(d):
            randomizer.seed(seeds[j])
            if set_lms:
                if print_scramble_mat: print('\n\ts[dvec[%d]]\n\t\t'%j,end='',flush=True)
                for t in range(t2_max):
                    t1 = min(t,t_max)
                    u = randomizer.randint(low=0, high=1<<t1, size=1, dtype=uint64)
                    u <<= (t_max-t1)
                    if t1<t_max: u += 1<<(t_max-t1-1)
                    for m in range(m_max):
                        v = u&zmsb[j,m]
                        s = self._count_set_bits(v)%2
                        if s: znew[j,m] += uint64(1<<(t2_max-t-1))
                    if print_scramble_mat:
                        for tprint in range(t_max):
                            mask = 1<<(t_max-tprint-1)
                            bit = (u&mask)>0
                            print('%-2d'%bit,end='',flush=True)
                        print('\n\t\t',end='',flush=True)
            else:
                znew = zmsb
            if set_rshift:
                rshift[j] = randomizer.randint(low=0, high=2**t2_max, size=1, dtype=uint64)
        return znew,rshift

    def _flip_bits(self, e, t_max):
        """
        flip the int e with t_max bits
        """
        u = 0
        for t in range(t_max):
            bit = array((1<<t),dtype=uint64)&e
            if bit:
                u += 1<<(t_max-t-1)
        return u

    def _count_set_bits(self, e):
        """
        count the number of bits set to 1 in int e
        Brian Kernighan algorithm code: https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
        """
        if (e == 0): return 0
        else: return 1 + self._count_set_bits(e&(e-1)) 

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True, return_unrandomized=False):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_unrandomized (bool): return the LMS matrix without digital shift. 
                Only applies when randomize='LMS' (the default). 

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_min == 0 and self.set_rshift==False and warn:
            warnings.warn("Non-randomized DigitalNetB2 sequence includes the origin",ParameterWarning)
        if return_unrandomized and not (self.set_lms and self.set_rshift):
            print(self.set_lms,self.set_rshift)
            raise ParameterError("return_unrandomized=True only applies when randomize=True.")
        n = int(n_max-n_min)
        x = zeros((n,self.d), dtype=double)
        xr = zeros((n,self.d),dtype=double)
        rc = self.dnb2_cf(n_min,n,self.d,self.graycode,self.m_max,self.t2_max,self.znew,self.set_rshift,self.rshift,x, xr)
        if rc!=0:
            raise ParameterError(self.errors[rc])
        if return_unrandomized:
            return xr,x
        elif self.set_rshift:
            return xr
        else:
            return x
    
    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[0], dtype=float)

    def set_seed(self, seed=None):
        if isinstance(seed,list) or isinstance(seed,ndarray):
            self.seeds = array(seed)
            seed = seed[0]
            super(DigitalNetB2,self).set_seed(seed)
        else:
            self.seed = seed
            super(DigitalNetB2,self).set_seed(seed)
            self.seeds = self.rng.choice(1000000, self.d, replace=False).astype(uint64)+1
        self.znew,self.rshift = self.set_digitalnetb2_randomizations(
            self.dvec,self.d,self.set_lms,self.set_rshift,self.seeds,self.d_max,self.m_max,
            self.t_max,self.t2_max,self.z,self.msb,self.verbose) 

    def _set_dimension(self, dimension, newseed=None):
        """
        Reset the dimension

        Args:
            dimension (int): new dimension
        """
        if isinstance(dimension,list) or isinstance(dimension,ndarray):
            self.dvec = array(dimension)
            self.d = len(self.dvec)
        else:
            self.d = dimension
            self.dvec = arange(self.d)
        self.seeds = self.rng.choice(1000000, self.d, replace=False).astype(uint64)+1
        self.znew,self.rshift = self.set_digitalnetb2_randomizations(
            self.dvec,self.d,self.set_lms,self.set_rshift,self.seeds,self.d_max,self.m_max,
            self.t_max,self.t2_max,self.z,self.msb,self.verbose)       

Sobol = DigitalNetB2
