import numpy as np
import cmath as z
import scipy as sp
import matplotlib.pyplot as plt
import simpleLinearSystem as example
import StuartLandau as example_sl
import myUtils
import algs

## Example selector
# 0 = simple linear model as implemented in matlab code of Takeishi's
# 1 = Stuart Landau model implementation
example_number = 0

# global setting for data point count
intDataCount=1000


######################

if example_number==0:
    example.run_simple_linear_example(intDataCount,noiseStdObs=0,noiseStdProcess=0.1)
elif example_number==1:                     #Run landau stuart example

    dictParas = {'mu':0.9,
                 'gamma':0.9,
                 'beta':0.9,
                 'dt':0.01,
                 'sigma_p':0.5,
                 'sigma_o':0.0}

    liInit = [0.0001,0.0001]  # [r, theta]

    data = example_sl.generate_stuart_landau(intDataCount,dictParas, liInit)
    myUtils.plot_polar_coords(data)
    observables = example_sl.generate_observables_matrix(data,10)

    Y0= observables[:,:-1]
    Y1= observables[:,1:]

    r=np.linalg.matrix_rank(Y0)
    print('Y0 mterics: rank=',r,' shape=',Y0.shape)
    lam_ord,_,_ = algs.ordDMD(Y0,Y1,r)

    lam_ord_vector_cart = [np.log(lam_ord[i,i])/dictParas['dt'] for i in range(lam_ord.shape[0])]
    myUtils.plot_point_array(lam_ord_vector_cart,'x','red')

#print ('eignevalues',[lam_ord[i,i] for i in range(lam_ord.shape[0])])


