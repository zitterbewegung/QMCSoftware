from qmcpy import *
from qmcpy.integrand.LR import LR
from numpy import *

# load data
n = 10
data = genfromtxt('binary.csv', dtype=float, delimiter=',', skip_header = True)
s = data[:n, 1:]
t = data[:n, 0]
no,dim_s = s.shape

r = dim_s +1
dim = r+1
lr = LR(Sobol(dim_s+1,seed=8), s_matrix = s, t = t, r = r, prior_variance=[1,1e-4,1,1])
#https://www.markdownguide.org/cheat-sheet/
#https://qmcpy.readthedocs.io/en/latest/demo_rst/elliptic-pde.html


# familiar method
'''qmcclt = CubQMCCLT(lr,abs_tol=1e-3,rel_tol=.1e-3)
s,data = qmcclt.integrate()
print(data,'\n\n'+'~'*100+'\n\n')'''

# new method
if r==0: raise Exception('require r>0')
qmcclt = CubQMCCLT(lr,
    abs_tol = 0,
    rel_tol = .25,
    n_init = 256,
    n_max = 2 ** 30,
    inflate = 1.2,
    alpha = 0.01,
    replications = 16,
    error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol),
    bound_fun = lambda phvl, phvh: (
        minimum.reduce([phvl[1:dim]/phvl[0],phvl[1:dim]/phvh[0],phvh[1:dim]/phvl[0],phvh[1:dim]/phvh[0]]),
        maximum.reduce([phvl[1:dim]/phvl[0],phvl[1:dim]/phvh[0],phvh[1:dim]/phvl[0],phvh[1:dim]/phvh[0]]),
        sign(phvl[0])!=sign(phvh[0])),
    dependency = lambda flags_comb: hstack((flags_comb.any(),flags_comb)))
s,data = qmcclt.integrate()
print(data)
