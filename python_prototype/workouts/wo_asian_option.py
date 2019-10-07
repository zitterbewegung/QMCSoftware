#! /usr/bin/env python_prototype
"""
Single Level Asian Option Pricing
    Run: python workouts/wo_asian_option.py
    Save Output: python workouts/wo_asian_option.py  > outputs/ie_AsianOption.txt
"""

from numpy import arange

from qmcpy import integrate
from qmcpy._util import summarize
from qmcpy.stop import CLT,CLTRep
from qmcpy.distribution import IIDDistribution,QuasiRandom
from qmcpy.integrand import AsianCall
from qmcpy.measures import StdUniform,StdGaussian,BrownianMotion,Lattice,Sobol

def test_distributions():
    # IID std_uniform
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdUniform(dimension=[64]), seed_rng=7)
    stopObj = CLT(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # IID std_gaussian
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdGaussian(dimension=[64]), seed_rng=7)
    stopObj = CLT(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom lattice
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Lattice(dimension=[64]),seed_rng=7)
    stopObj = CLTRep(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom sobol
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Sobol(dimension=[64]), seed_rng=7)
    stopObj = CLTRep(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

    """ Multi-Level Asian Option Pricing """
    # IID std_uniform
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdUniform(dimension=[4, 16, 64]), seed_rng=7)
    stopObj = CLT(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # IID std_gaussian
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdGaussian(dimension=[4, 16, 64]), seed_rng=7)
    stopObj = CLT(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom Lattice
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Lattice(dimension=[4,16,64]),seed_rng=7)
    stopObj = CLTRep(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom sobol
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Sobol(dimension=[4, 16, 64]), seed_rng=7)
    stopObj = CLTRep(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    summarize(stopObj,measureObj,OptionObj,distribObj,dataObj)

test_distributions()
