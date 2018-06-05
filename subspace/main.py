import numpy as np
import cmath as z
import scipy as sp
import matplotlib.pyplot as plt
import simpleLinearSystem as example
import StuartLandau as example_sl
import myUtils
import algs

# simple example. Run using defaults.
#example.run_simple_linear_example()

intDataCount=10000
dictParas = {'mu':0.7,
             'gamma':0.9,
             'beta':0.9,
             'dt':0.01,
             'sigma_p':0.5,
             'sigma_o':0.05}

liInit = [0.001,0.001]  # [r, theta]

data = example_sl.generate_stuart_landau(intDataCount,dictParas, liInit)
#myUtils.plot_polar_coords(data)

Y0= example_sl.generate_observables_matrix(data[:,:-1],10)
Y1= example_sl.generate_observables_matrix(data[:,1:],10)

r=np.rank(Y0)
lam_ord,_,_ = algs.ordDMD(Y0,Y1,r)
print (lam_ord)


