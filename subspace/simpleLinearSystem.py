import numpy as np
import algs
import myUtils
import scipy as sp


def generateData2D(intDataCount, eigs, noiseStdProc=0, test = False):
    """
    Generate data for simple example : 2 dimensional dynamics on complex plane and optionally add process noise
    Inputs:
    ------
    intDataCount is number of data points to genereate
    eigs : list 1x2 : true eigenvalues of the linear dynamics system
    noiseStd : std of process noise to add
    test: bool : test==true then add process noise at each step.
    returns
    -------
    Y_proc : 2 x intDataCount array : complex valued time series of linear dynamical system
    """

    Y = np.zeros((len(eigs), intDataCount), dtype=np.complex64)  # 2x 1000
    Y[:, :1] = np.ones([len(eigs), 1])  # zeroth column

    for index in range(1, intDataCount):
        if not test:
            Y[:, index:(index + 1)] = np.matmul(sp.diag(eigs), Y[:, (index - 1):index]) + np.random.randn(2, 1) * noiseStdProc
        else:
            Y[:, index:(index + 1)] = np.matmul(sp.diag(eigs), Y[:, (index - 1):index]) + np.matrix([index, index]).T * noiseStdProc

    return Y

def run_simple_linear_example(intDataCount = 1000, noiseStdProcess = 0.1,noiseStdObs = 0.1, r = 0.9, test=False):
    """"
    Simple 2x2 linear systems implemented in Matlab example by Takaishi
    2D linear system with eignevalues of dynamics being +/-i
    inputs
    ------
    noiseStdxx: float : scale factor of standard normal noise
    r : float : is modulus of eigenvalue
    noiseStdx : float : noise magnitude. Assumed noise is standard normal * noise mag.
    """
    
    # Generate data ##############

    # Define the true eigenvalue of example linear system to be +/- i
    lam_true = [r * np.exp(complex(0, 1) * np.pi / 180 * 90), r * np.exp(complex(0, 1) * np.pi / 180 * (-90))]

    Y_noiseless = generateData2D(intDataCount, lam_true, 0,test)  # noiseless data generated
    Y_proc = generateData2D(intDataCount, lam_true, noiseStdProcess,test)  # data with process noise

    if not test:
        obs_noise = np.random.randn(intDataCount)*noiseStdObs   # data containing both proc and observation noise.
    else:
        obs_noise = np.linspace(1,3,intDataCount)*noiseStdObs   # simple deterministic data for testing.

    Y_obs_proc = Y_proc + obs_noise

    # Show raw data
    myUtils.plot_raw_data( Y_obs_proc, Y_noiseless)
  
    # Run DMD algorithms #######

    # Do exact DMD
    rank = np.linalg.matrix_rank(Y_obs_proc)
    lam_ord, v, w = algs.ordDMD(Y_obs_proc[:, 0:(intDataCount-1)], Y_obs_proc[:, 1:intDataCount], rank)

    # Do Subspace DMD
    lam_sub = algs.subspaceDMD(Y_obs_proc)[0]

    # Print eigenvalues
    print('True Eigenvalues',lam_true)
    print('Ord DMD eigenvalues',lam_ord)
    print ('Subspace DMD eigenvalues', lam_sub)

    # Plot in cartesian co-ordinates
    myUtils.plot_eigs_cart(lam_true, lam_ord, lam_sub)