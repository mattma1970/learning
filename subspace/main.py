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


#Run landau stuart example

intDataCount=10000
dictParas = {'mu':0.9,
             'gamma':0.9,
             'beta':0.9,
             'dt':0.01,
             'sigma_p':0.5,
             'sigma_o':0.05}

liInit = [0.001,0.001]  # [r, theta]

data = example_sl.generate_stuart_landau(intDataCount,dictParas, liInit)
#myUtils.plot_polar_coords(data)
observables = example_sl.generate_observables_matrix(data,10)

Y0= observables[:,:-1]
Y1= observables[:,1:]

r=np.linalg.matrix_rank(Y0)
print('Y0 mterics: rank=',r,' shape=',Y0.shape)
lam_ord,_,_ = algs.ordDMD(Y0,Y1,r)

lam_ord_vector_cart = [lam_ord[i,i] for i in range(lam_ord.shape[0])]
myUtils.plot_point_array(lam_ord_vector_cart,'x')

#print ('eignevalues',[lam_ord[i,i] for i in range(lam_ord.shape[0])])


