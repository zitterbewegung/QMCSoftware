from qmcpy.integrand.LR import LR
from sys import meta_path
import numpy
from qmcpy import *

def lowerbound(den, num, error_den, error_num):
    lower_array = numpy.zeros((len(num), 4), dtype = float)
    for i in range(len(num)):
        test1 = (num[i] - error_num[i]) / (den - error_den)
        test2 = (num[i] - error_num[i]) / (den + error_den)
        test3 = (num[i] + error_num[i]) / (den - error_den)
        test4 = (num[i] + error_num[i]) / (den + error_den)
        lower_array[i]= min(test1, test2, test3, test4)
    return lower_array

def upperbound(den, num, error_den, error_num):
    upper_array = numpy.zeros((len(num)), dtype = float)
    for i in range(len(num)):
        test1 = (num[i] - error_num[i]) / (den - error_den)
        test2 = (num[i] - error_num[i]) / (den + error_den)
        test3 = (num[i] + error_num[i]) / (den - error_den)
        test4 = (num[i] + error_num[i]) / (den + error_den)
        upper_array[i]= max(test1, test2, test3, test4)
    return upper_array

def estimate(lowbound, highbound):
    return (lowbound + highbound) / 2

data = numpy.genfromtxt('binary.csv', dtype=float, delimiter=',', skip_header = True)

n = 10
#https://stats.idre.ucla.edu/r/dae/logit-regression/
s = data[:n, 1:]
t = data[:n, 0]

no, dim = s.shape

prior_variance = [1,1e-4,1,1]

"""
k = LR(IIDStdUniform(dim+1,seed=8), s_matrix = s, t = t)
solution, data = CubMCCLT(k, abs_tol = .001).integrate()
print(data)
print(" ")
k1 = LR(Sobol(dim+1,seed=8), s_matrix = s, t = t,  prior_variance = prior_variance)
solution1, data1 = CubQMCSobolG(k1, abs_tol = .001).integrate()
print(data1)
print(" ")

my_instance = LR(sampler = Sobol(dimension=dim+1, seed = 7), s_matrix = s, t = t)
p = my_instance.discrete_distrib.gen_samples(n_min=0, n_max=1024)
y = my_instance.f(p)
print(y.mean())
"""
k = LR(Sobol(dim+1,seed=8), s_matrix = s, t = t, r = 3, prior_variance = prior_variance)
solution, data = CubQMCCLT(k, abs_tol = 0, rel_tol = .1).integrate()
print(data)
print(data.error_bound)
print(" ")
print(solution)
den = solution[0]
num = solution[1:]
e_den = data.error_bound[0]
e_num = data.error_bound[1:]
lowbound = lowerbound(den, num, e_den, e_num)
highbound = upperbound(den, num, e_den, e_num)
print(lowbound)
print(highbound)
print(estimate(lowbound, highbound))