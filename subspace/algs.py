import numpy as np
import cmath as z
import scipy as sp


def reducedSVD(*args):
    """
    performs a reduced rank SVD on the passed in matrix X=USV^h
    inputs
        X : data matrix with data per column
        r : desired rank (opt)
    returns:
        U, S, Vh of  singluar decomp. Where Vh is hermitian conj.
    """
    X=args[0]
    if len(args)==1:
        r= np.linalg.matrix_rank(X)
    else:
        r=args[1]
            
    U_r, S_r, V_r = np.linalg.svd(X, full_matrices=False) #full_matrices=False is equivalent of matlab 'econ' setting for SVD.

    U_r = U_r[:,0:r]
    S_r=S_r[0:r]
    V_r=V_r[0:r,:]

    print('SVD Check. USV==Data is ',np.allclose(X,U_r@np.diag(S_r)@(V_r)))
    err = X-U_r@np.diag(S_r)@(V_r)
    err = np.array(err)@(np.array(err).conj().T)
    print ('reconstruction error (Frob Norm)',np.sqrt(sum([err[i,i] for i in range(err.shape[0])])))

    return U_r,S_r,V_r



def ordDMD(Y0, Y1, r):
    """ 
    Ordinary exact DMD (no augmentation via observation functions)
    Inputs:
        Y0,Y1 : np.arrays : data snapshots for t=0 and then time shifted
        r: rank of desired reduced SVD approximation
    Returns:
        lam : rxr np.array : eigenvalues
        v_right :   : normalised right eigenvector
        w :  : modes of the dynamics 
    """

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



def subspaceDMD(Y, r=-1):
    """
    subspace DMD . A DMD algorithm proportedly robust to process and input noise.
    Inputs:
        Y: np.array: r x intDataCount array of noisey data
        r: int : desired rank of svd
    Returns: 
        lam: rxr array: diagonal eigenvalue matrix
    """
    
    # Step 1: create the past and future data sets
    Y0 = Y[:, 0:-3]
    Y1 = Y[:, 1:-2]
    Y2 = Y[:, 2:-1]
    Y3 = Y[:, 3:]

    [n, m] = Y0.shape

    Yp = np.vstack((Y0, Y1))
    Yf = np.vstack((Y2, Y3))

    # step 2: Compute the orthogonal projection of Yf onto Yp.H
    _, _, Vp = reducedSVD(Yp)
    O = (Yf @ Vp.conj().T) @ Vp
    
    if r==-1:       # if no r value passed in use the full rank of projection
        rank = np.linalg.matrix_rank(O)
    else:
        rank = r
    rank = np.min([n,rank])     # cap rank at dimensionality of passed in times series
  
    # step 3: Define Uq1 an Uq2
    Uq, _, _ = reducedSVD(O)
    Uq1 = Uq[0:n, :rank]
    Uq2 = Uq[n:, :rank]
    
    # step 4: Compute the SVD of Uq1
    U1, S1, Vh1 = reducedSVD(Uq1)
    M = Uq2 @ Vh1.conj().T @ np.diag([1. / i for i in S1])
    A_tilde = U1.conj().T @ M
    
    # step 5: Eigendecomp of A_tilde
    lam, w_tilde = np.linalg.eig(A_tilde)
    # step 6: return the modes
    w = M @ w_tilde @ np.diag([1. / i for i in lam])
    lam = np.diag(lam)

    return lam, w
