import numpy as np
import cmath as z
import scipy as sp
import numpy.random as rnd


def generate_stuart_landau(intDataCount, dictParas, liInit):
    # Generates data from stuart landau equations
    # intDataCount : number of data points to generate
    # dictPara : dictionary of parameters,namely, mu,gamma,beta,dt and sigma_p
    # liInit: list of initial values [r, theta]

    data=np.zeros((2,intDataCount),dtype=complex)  # instantiate data array
    data[:,0]=np.array(liInit)  # 2x1 matrix of initial conditions

    for counter in np.arange(1,intDataCount):
         r_t = data[0,counter-1];
         theta_t=data[0,counter-1];

         # discretized stuart-landau solution
         temp = np.array([r_t+(dictParas['mu']*r_t-r_t**3)*dictParas['dt'],
                            theta_t + (dictParas['gamma']-dictParas['beta']*r_t**2)*dictParas['dt']])
         temp2 = temp + np.diag([dictParas['dt'],dictParas['dt']/r_t])@(rnd.multivariate_normal([0,0],[[1,0],[0,1]])*dictParas['sigma_p'])

         data[:,counter] = np.array(temp2)

    return data

# Generate the extended data matrix using
def generate_observables_matrix(data,n,obs_noise_std=0.05,bAddNoise=True):
    # data: full data matrix
    # n: 1/2 the number of observation functions to use (2n+1)
    # bAddNoise: boolean indicating if additive noise should be introduced.

    obs = np.zeros((2*n+1,data.shape[1]),dtype=complex)

    for column_num in range(data.shape[1]):
        o = np.array([np.exp(m * complex(0, 1)*data[1,column_num]) for m in np.arange(-n, n+1, 1)])
        o2= np.array([rnd.normal(0,1) for i in range(2*n+1)])

        obs[:, column_num]=o+o2

    return obs














