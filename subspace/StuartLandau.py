import numpy as np
import cmath as z
import scipy as sp
import numpy.random as rnd


def generate_stuart_landau(intDataCount, dictParas, liInit):
    """
    Generates data from stuart landau equations with optional process noise
    Note that observation noise is implemented in gerneate_observables_matrix
    inputs
    ======
        intDataCount : int : number of data points to generate
        dictPara : dictionary of parameters,namely, mu,gamma,beta,dt and sigma_p
        liInit: list : initial values [r, theta]
    returns
    =======
        npArray : 2 x intDataCount : simulated data from stuart landau equation
        
    """

    data=np.zeros((2,intDataCount)) # instantiate data array
    data[:,0]=np.array(liInit)  # 2x1 matrix of initial conditions

    for counter in np.arange(1,intDataCount):
        r_t = data[0,counter-1];
        theta_t=data[1,counter-1];

        # discretized stuart-landau solution
        temp = np.array([r_t+(dictParas['mu']*r_t-r_t**3)*dictParas['dt'],
                            theta_t + (dictParas['gamma']-dictParas['beta']*r_t**2)*dictParas['dt']]).reshape(2,1)
        temp2 = temp + np.diag([dictParas['dt'],dictParas['dt']/r_t])@(rnd.randn(2,1)*dictParas['sigma_p'])

        data[:,counter] = np.array(temp2.reshape(2,))

    return data


# Generate the extended data matrix using
def generate_observables_matrix(data,n,dictParas):
    """
    Generate observable matrix using complex exponentials
    Inputs
    ======
        data: np.array : full data matrix
        n: int: 1/2 the number of observation functions to use (2n+1)
        dictParas: dictionary containing observation noise value
    returns
    =======
        abs: nparray of 2n+1 x intDataCount : observation noise corrupted observables matrix
    
    """

    obs = np.zeros((2*n+1,data.shape[1]),dtype=complex)

    for column_num in range(data.shape[1]):
        o = np.array([complex(np.cos(m*data[1,column_num]),np.sin(m*data[1,column_num])) for m in np.arange(-n, n+1, 1)]).reshape(2*n+1,1)
        o2= rnd.randn(2*n+1,1)*dictParas['sigma_o']

        obs[:, column_num:(column_num+1)]=(o+o2)

    return obs














