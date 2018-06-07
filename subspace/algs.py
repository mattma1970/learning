import numpy as np
import cmath as z
import scipy as sp


# performs a reduced rank SVD on the passed in matrix X=USV^h
def reducedSVD(X,r):
    # X : data matrix with data per column
    # r : desired rank
    U_r, S_r, V_r = np.linalg.svd(X)

    U_r = U_r[:,0:r]
    S_r=S_r[0:r]
    V_r=V_r[0:r,:]

    print(np.allclose(X,U_r@np.diag(S_r)@(V_r)))

    return U_r,S_r,V_r


# ordinary exact DMD (no augmentation via observation functions)
def ordDMD(Y0, Y1, r):
    # Inputs:
    #   Y0,Y1 : np.arrays : data snapshots for t=0 and then time shifted
    #   r: rank of desired reduced SVD approximation
    # Returns:
    #   lam : rxr np.array : eigenvalues
    #   v_right :   : normalised right eigenvector
    #   w :  : modes of the dynamics

    ## Exact DMD

    # step 1 : perform SVD with reduced rank r
    U_r, S_r, Vh_r = reducedSVD(Y0, r)
    # step 2: projection onto POD
    M = Y1 @ Vh_r.conj().T @ np.diag([1. / i for i in S_r])
    A_tilde = U_r.conj().T @ M
    # step 3 eigen decomp of a_tilde
    lam, v_right = np.linalg.eig(A_tilde)
    # step 4: calculate the modes
    w = M @ v_right @ np.diag(1. / lam)

    return np.diag(lam), v_right, w


# subspace DMD . A DMD algorithm proportedly robust to process and input noise.
def subspaceDMD(Y, r):
    # Inputs:
    # Y_proc: np.array: 2 x intDataCount or observations corrupted by observation noise

    paraCount = len(locals())

    # create the past and future data sets
    Y0 = Y[:, 0:-3]
    Y1 = Y[:, 1:-2]
    Y2 = Y[:, 2:-1]
    Y3 = Y[:, 3:]

    [n, m] = Y0.shape

    Yp = np.vstack((Y0, Y1))
    Yf = np.vstack((Y2, Y3))

    # step 2: Compute the orthogonal projection of Yf onto Yp.H
    _, _, Vp = np.linalg.svd(Yp)
    O = (Yf @ Vp) @ Vp.conj().T

    if paraCount == 1:
        rank = np.linalg.matrix_rank(O)
    else:
        rank = r
    # print(rank)

    # step 3: Defined Uq1 an Uq2
    Uq, _, _ = reducedSVD(O, r)
    Uq1 = Uq[0:n, 0:rank]
    Uq2 = Uq[n:, 0:rank]
    # step 4: Compute the SVD of Uq1
    U1, S1, Vh1 = np.linalg.svd(Uq1)
    M = Uq2 @ Vh1.conj().T @ np.diag([1. / i for i in S1])
    A_tilde = U1.conj().T @ M
    # step 5: Eigendecomp of A_tilde
    lam, w_tilde = np.linalg.eig(A_tilde)
    # step 6: return the modes
    w = M @ w_tilde @ np.diag([1. / i for i in lam])
    lam = np.diag(lam)

    return lam, w
