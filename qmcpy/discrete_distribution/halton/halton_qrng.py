from ..c_lib import c_lib
import ctypes
from numpy import *
from ...util import ParameterError, ParameterWarning
import warnings


class HaltonQRNG(object):
    """
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """
    def __init__(self, dimension, generalize, randomize, seed)):
        self.halton_qrng_cf = c_lib.halton_qrng
        self.halton_qrng_cf.argtypes = [
            ctypes.c_int,  # n
            ctypes.c_int,  # d
            ctypes.c_int,  # generalized
            numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
            ctypes.c_long]  # seed
        self.halton_qrng_cf.restype = None
        self.r = randomize
        if self.randomize==False:
            raise ParameterError('QRNG Halton must be randomized. Use Owen backend for non-randomized Halton.', ParameterWarning)
        self.d_lim = 360
        self.n_lim = 2**32-1
        self.set_dimension(dimension)
        self.set_seed(seed)

    def gen_samples(self, n_min=0, n_max=8, warn=True):
        if n_min > 0:
                raise ParameterError('QRNG Halton does not support skipping samples. Use Owen backend to set n_min > 0.')
        if n_max > self.n_lim:
            raise ParameterWarning("QRNG Halton requres n_max < 2^32-1.")
        n = int(n_max-n_min)
        x = zeros((d, n), dtype=double)
        self.halton_qrng_cf(n, d, generalize, x, seed)
        return x.T
    
    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32)
        
    def set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError('QRNG Halton requires dimension <= %d'%self.d_lim)
    