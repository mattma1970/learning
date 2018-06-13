import numpy as np
import cmath as z
import scipy as sp
import matplotlib.pyplot as plt
import simpleLinearSystem as example
import StuartLandau as example_sl
import myUtils
import algs

"""
Example selector
    0 = simple linear model as implemented in matlab code of Takeishi's
    1 = Stuart Landau model implementation
"""
example_number = 1

# global setting for data point count
intDataCount=10000


if example_number==0:
    example.run_simple_linear_example(intDataCount,noiseStdObs=0.1,noiseStdProcess=0.1,test=False)
elif example_number==1:      
    # Run the stuart landau equation example 
    
    """ No noise case """
    # Dictionary of parameters for Stuart Landau equation
    dictParas = {'mu':1,
                 'gamma':1,
                 'beta':0,
                 'dt':0.01,
                 'sigma_p':0.0,
                 'sigma_o':0.0}
    # Initial values of r and theta respectively 
    liInit = [2,0]
    
    # Generate the data
    data = example_sl.generate_stuart_landau(intDataCount,dictParas, liInit)
    observables = example_sl.generate_observables_matrix(data,10,dictParas)
    
    # Prepare data for ordinary DMD that required time displaced data sets.
    Y0= observables[:,:-1]
    Y1= observables[:,1:]
    r=np.linalg.matrix_rank(Y0)
    lam_ord,_,_ = algs.ordDMD(Y0,Y1,r)
    # and then plot the results.
    lam_ord_vector_cart = [np.log(lam_ord[i,i])/dictParas['dt'] for i in range(lam_ord.shape[0])]
    fig_orig= myUtils.plot_point_array(lam_ord_vector_cart,'x','blue',blSetXScale=True, x_range_scale=0.05,title='Noiseless data Ord DMD eigenvalues')
    
    """ Process and obs noise cases """
    # Dictionary of parameters for Stuart Landau equation
    dictParas = {'mu':1,
                 'gamma':1,
                 'beta':0,
                 'dt':0.01,
                 'sigma_p':0.3,
                 'sigma_o':0.01}
    
    # Initial values of r and theta respectively 
    liInit = [2,0]
    
    # Generate the data
    data = example_sl.generate_stuart_landau(intDataCount,dictParas, liInit)
    myUtils.plot_polar_coords(data) # assert that it falls into a periodic cycle or r=1
   
    # generate the observable using trigonometric functions
    observables = example_sl.generate_observables_matrix(data,10,dictParas)
    
    # Prepare data for ordinary DMD that required time displaced data sets.
    Y0= observables[:,:-1]
    Y1= observables[:,1:]
    r=np.linalg.matrix_rank(Y0)
    print('Y0 mterics: rank=',r,' shape=',Y0.shape)
    lam_ord,_,_ = algs.ordDMD(Y0,Y1,r)
    # and then plot the results.
    lam_ord_vector_cart = [np.log(lam_ord[i,i])/dictParas['dt'] for i in range(lam_ord.shape[0])]
    fig1 = myUtils.plot_point_array(lam_ord_vector_cart,'x','red',blSetXScale=True, x_range_scale=0.03,title='ordinary DMD eigenvalues', chainToAxes=fig_orig)
    
    # Apply the subspace DMD algorithm
    lam_sub = algs.subspaceDMD(observables)[0]
    lam_sub_cart=[np.log(lam_sub[i,i])/dictParas['dt'] for i in range(lam_sub.shape[0])]
    fig2 = myUtils.plot_point_array(lam_sub_cart,'+','green',blSetXScale=True ,x_range_scale=0.03,title='subSpaceDMD eigenvalues', chainToAxes=fig1)


    # Display the chart will all methods for comparison
    fig2.set_title('DMD Methods')
    fig2.legend()
    plt.show()
    



