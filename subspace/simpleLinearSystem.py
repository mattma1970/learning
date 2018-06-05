import numpy as np
import algs
import myUtils

def run_simple_linear_example(intDataCount = 1000, noiseStdProcess = 0.05,noiseStdObs = 0.1, r = 0.9):

    ## Debug parameter
    test=False

    ########## Generate data ##############

    # true eigenvalue of example linear system
    lam_true = [r * np.exp(complex(0, 1) * np.pi / 180 * 90), r * np.exp(complex(0, 1) * np.pi / 180 * (-90))]

    Y_noiseless = myUtils.generateData2D(intDataCount, lam_true, 0,test)  # noiseless data generated
    Y_proc = myUtils.generateData2D(intDataCount, lam_true, noiseStdProcess,test)  # data with process noise

    if not test:
        obs_noise = np.random.randn(intDataCount)*noiseStdObs   # data containing both proc and observation noise.
    else:
        obs_noise = np.linspace(1,3,intDataCount)*noiseStdObs

    Y_ob_proc = Y_proc + obs_noise

    # Show raw data
    myUtils.plot_raw_data(Y_proc, Y_noiseless)
    myUtils.plot_raw_data(Y_proc,Y_ob_proc)

    ########### Run DMD algorithms #######

    # Do exact DMD
    rank = np.linalg.matrix_rank(Y_ob_proc)
    lam_ord, v, w = algs.ordDMD(Y_ob_proc[:, 0:(intDataCount-1)], Y_ob_proc[:, 1:intDataCount], rank)

    # Do Subspace DMD
    lam_sub = algs.subspaceDMD(Y_ob_proc,rank)[0]

    # Print eigenvalues
    print('True Eigenvalues',lam_true)
    print('Ord DMD eigenvalues',lam_ord)
    print ('Subspace DMD eigenvalues', lam_sub)

    # Plot in cartesian co-ordinates
    myUtils.plot_eigs_cart(lam_true, lam_ord, lam_sub)