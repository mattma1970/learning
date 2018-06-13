import numpy as np
import cmath as z
import scipy as sp
import matplotlib.pyplot as plt




# convert to polar and stack into matrix
def convertPolar(Y):
    # Y_proc : np.array : 2 x intDataCount
    # ret : array: polar co-ordinates radius and phase of shape 4 x 1000
    ret = []
    dims = Y.ndim
    counter = Y.shape[0]
    if dims == 1:
        counter = 1  # if vector reset counter

    for i in range(0, counter):
        r = [abs(pt) for pt in np.nditer(Y[:])]
        theta = [z.phase(pt) for pt in np.nditer(Y[:])]
        ret.append(r)
        ret.append(theta)
    return np.array(ret)


# convert to cartesian and stack into matrix
def convertCart(Y):
    # Y : np.array : 2 x intDataCount
    # ret : array: [y; x]
    ret = []
    dims = Y.ndim
    counter = Y.shape[0]
    if dims == 1:
        counter = 1  # if vector reset counter

    for i in range(0, counter):
        r = [pt.real for pt in np.nditer(Y[:])]
        im = [pt.imag for pt in np.nditer(Y[:])]
        ret.append(im)
        ret.append(r)
    return np.array(ret)


def plot_raw_data(Y, Y_noiseless):
    fig = plt.figure(figsize=(20,20))
    ax=fig.add_subplot(221,polar=True)
    ax2=fig.add_subplot(223,polar=True)
    ax3=fig.add_subplot(224,polar=True)
    ax4=fig.add_subplot(222,polar=True)

    ax.set_title("Coord 1 or noiseless dynamics")
    ax2.set_title("Coord 1 evolution in time under noisey dynamics")
    ax3.set_title("Coord 2 evolution in time under noisey dynamics")
    ax4.set_title("Coord 2 - Noiseless dynamics")
    line,=ax.plot(convertPolar(Y_noiseless[0])[1],convertPolar(Y_noiseless[0])[0], color='green',lw=1)
    line2,=ax2.plot(convertPolar(Y[0])[1],convertPolar(Y[0])[0],lw=1,color='red')
    line3,=ax3.plot(convertPolar(Y[1])[1],convertPolar(Y[1])[0],lw=1,color='red')
    line4,=ax4.plot(convertPolar(Y_noiseless[1])[1],convertPolar(Y_noiseless[1])[0],lw=1,color='green')

    plt.show()
    return

def plot_polar_coords(Y):
    # plot array of polar co-ordinates on a polar plot
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    line=ax.plot(Y[1,:],Y[0,:],lw=1,color='red')
    #plt.show()


def plot_eigs_polar(lam_true, lam_ord, lam_sub):
    # get data in polar coordinates for plotting
    ord_polar = convertPolar(np.diagonal(lam_ord))
    sub_polar = convertPolar(np.diagonal(lam_sub))
    true_polar = convertPolar(np.array(lam_true))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    for i in range(0, 2):
        ax.plot([true_polar[1, i]], [true_polar[0, i]], 'o')
        ax.annotate('true', xy=(true_polar[1, i], true_polar[0, i]))
        ax.plot([ord_polar[1, i]], [ord_polar[0, i]], 'x')
        ax.annotate('ordinary DMD', xy=(ord_polar[1, i], ord_polar[0, i]))
        ax.plot([sub_polar[1, i]], [sub_polar[0, i]], '+')
        ax.annotate('subscpace DMD', xy=(sub_polar[1, i], sub_polar[0, i]))

    plt.show()
    return


def plot_eigs_cart(lam_true, lam_ord, lam_sub):

    ord_c = convertCart(np.diagonal(lam_ord))
    sub_c = convertCart(np.diagonal(lam_sub))
    true_c = convertCart(np.array(lam_true))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for i in range(0, 2):
        ax.plot([true_c[1, i]], [true_c[0, i]], 'o')
        ax.annotate('true', xy=(true_c[1, i], true_c[0, i]))
        ax.plot([ord_c[1, i]], [ord_c[0, i]], 'x')
        ax.annotate('ordinary DMD', xy=(ord_c[1, i], ord_c[0, i]))
        ax.plot([sub_c[1, i]], [sub_c[0, i]], '+')
        ax.annotate('subspace DMD', xy=(sub_c[1, i], sub_c[0, i]))
        ax.plot(np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),"--")

    ax.set_xticks(np.arange(-1.1, 1.1, 0.1))
    ax.set_yticks(np.arange(-1.1, 1.1, 0.1))
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Eigenvalues of DMD methods")
    ax.grid()
    plt.show()

    return


def plot_point_array(data,marker='+',marker_color='green',blSetXScale=False, x_range_scale=1, title='chart',chainToAxes=None):
    """
    inputs
    ======
    Create figure and plot a list of complex values on cartesian axes.
    chainToAxes: fig.axes object : used to chain charts together in order have multiple data series on the same chart.
    returns
    =======
    plt.axes object containing the data from one or more data series.

    ** Does not call plt.show(). This must be done by the invoking method.

    """
    if type(data) is list:
        data=np.array(data,dtype=complex)

    if chainToAxes==None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_title(title)

        if blSetXScale==True:
            ax.set_autoscaley_on(True)
            ax.set_autoscalex_on(False)
            ax.set_xlim(-1*x_range_scale,1*x_range_scale)
        else:
            ax.set_autoscaley_on(True)
            ax.set_autoscalex_on(True)
            ax.autoscale()

        #ax.set_xticks(np.arange(-1.1, 1.1, 0.1))
        #ax.set_yticks(np.arange(-1.1, 1.1, 0.1))
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.grid()
    else:
        ax = chainToAxes

    dataCartReal = [data[i].real for i in range(data.shape[0])]
    dataCartImag = [data[i].imag for i in range(data.shape[0])]

    ax.scatter(dataCartReal,dataCartImag,color=marker_color, label=title)
    #ax.scatter(data[i].real, data[i].imag, marker,color=marker_color, label=title)

    return ax