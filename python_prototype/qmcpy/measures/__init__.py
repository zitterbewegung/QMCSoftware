""" Definition for abstract class Measure"""

from abc import ABC
from numpy import array, ndarray

from .._util import DistributionCompatibilityError, univ_repr

class Measure(ABC):
    """
    Specifies the components of a general measure used to define an
    integration problem or a sampling method
    """

    def __init__(self, dimension=None, **kwargs):
        """
        Construct a list of measures.

        Args:
            dimension (list of ints): Dimensions to be dispersed among list \
                of ``Measures``.
            **kwargs (dictionary): Accepts keyword arguments into dictionary. \
                Disperses dictionary values among list of ``Measures``
        """
        self.dimension = dimension
        super().__init__()
        if not dimension: return
        # Type check dimension
        if type(self.dimension)==int: self.dimension = array([self.dimension])
        if all(type(i)==int and i>0 for i in self.dimension): self.dimension = array(self.dimension)
        else: raise DimensionError("dimension must be an numpy.ndarray (or list) of positive integers")
        # Type check measureData
        for key,val in kwargs.items():
            if type(kwargs[key])!= list and type(kwargs[key])!= ndarray:
                kwargs[key] = [kwargs[key]]
            if len(kwargs[key]) == 1 and len(self.dimension)!=1:
                kwargs[key] = kwargs[key]*len(self.dimension)
            if len(kwargs[key]) != len(self.dimension):
                raise MeasureDataError(key+" must be a numpy.ndarray (or list) of len(dimension)")
        self.measure_list = [type(self)() for i in range(len(self.dimension))]
        # Create list of measures with appropriote dimensions and keyword arguments
        for i in range(len(self)):
            self[i].dimension = self.dimension[i]
            for key,val in kwargs.items():
                setattr(self[i],key,val[i])

    def __len__(self): return len(self.measure_list)
    def __iter__(self):
        for measureObj in self.measure_list: yield measureObj
    def __getitem__(self, i): return self.measure_list[i]
    def __setitem__(self, i, val): self.measure_list[i] = val
    def __repr__(self): return univ_repr(self)

    def summarize(self):
        h1 = '%s (%s)\n'
        item_s = '%25s: %-15s'
        s = ""

        s += h1 % (type(self).__name__, 'Measure Object')
        s += item_s % ('dimension', str(self.dimension))
        print(s)

# API
from .measures import StdUniform,StdGaussian,IIDZeroMeanGaussian,BrownianMotion,Lattice,Sobol


