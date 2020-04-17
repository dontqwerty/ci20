#Filename: HW1_skeleton.py

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

import seaborn as sns
from scipy.stats import expon
from numpy.linalg import inv
import math
from copy import copy, deepcopy
import statistics

scenario = 0
GN_mod = False # Used for 3.2: if True neglect the exp. distr. anchor to analyze differences

#--------------------------------------------------------------------------------
# Assignment 1
def main():
    global scenario
    # choose the scenario
    #scenario = 1    # all anchors are Gaussian
    #scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    scenario = 3    # all anchors are exponential
    
    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
    nr_anchors = np.size(p_anchor,0)
    
    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])
#    p_true = np.array([[2,-4])
                       
    #plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)
    
    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)
    
    # get the number of measurements 
    assert(np.size(data,0) == np.size(reference_measurement,0))
    nr_samples = np.size(data,0)
    
    #1) ML estimation of model parameters
    #TODO 
    params = parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref)
    
    #2) Position estimation using least squares
    #TODO
    position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, True)

    if(scenario == 3):
        # TODO: don't forget to plot joint-likelihood function for the first measurement

        #3) Postion estimation using numerical maximum likelihood
        #TODO
        position_estimation_numerical_ml(data,nr_anchors,p_anchor, params, p_true)
    
        #4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        #TODO
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov, params, p_true)

    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """
    params = np.zeros([1, nr_anchors])
    #TODO (1) check whether a given anchor is Gaussian or exponential
    for i in reference_measurement.T :
        ax = sns.distplot(i,
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
        ax.set(xlabel='Given Distribution', ylabel='Frequency')
        plt.show()
    #TODO (2) estimate the according parameter based
    t_reference_measurement = np.transpose(reference_measurement)
    t_reference_measurement_size = np.size(t_reference_measurement[0],0)
    if (scenario == 1) :
        params = np.array([np.cov(t_reference_measurement[0]),
                        np.cov(t_reference_measurement[1]),
                        np.cov(t_reference_measurement[2]),
                        np.cov(t_reference_measurement[3])])
    elif (scenario == 2) :
        params = np.array([t_reference_measurement_size / sum(t_reference_measurement[0]),
                        np.cov(t_reference_measurement[1]),
                        np.cov(t_reference_measurement[2]),
                        np.cov(t_reference_measurement[3])])
    elif (scenario == 3) :
        params = np.array([t_reference_measurement_size / sum(t_reference_measurement[0]),
                        t_reference_measurement_size / sum(t_reference_measurement[1]),
                        t_reference_measurement_size / sum(t_reference_measurement[2]),
                        t_reference_measurement_size / sum(t_reference_measurement[3])])
    print(params)
    return params
#--------------------------------------------------------------------------------
def position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, use_exponential):
    """estimate the position by using the least squares approximation. 
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        p_true... true position (needed to calculate error) 2x2 
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""
    nr_samples = np.size(data,0)
    
    #TODO set parameters
    tol = 0.0001 # valid, tested
    max_iter = 8 # valid, tested

    results = np.zeros((2000, 2)) # for collecting the estimated points
    errors = [] # for collecting the difference between estimated points and p_true
    p_start = np.random.uniform(np.min(p_anchor), np.max(p_anchor), 2)
    
    # TODO estimate position for  i in range(0, nr_samples)
    for i in range(2000):
        results[i] = least_squares_GN(p_anchor,p_start, data[i], max_iter, tol)
        errors.append(math.sqrt((p_true[0][0] - results[i][0])**2 + (p_true[0][1] - results[i][1])**2))

	# TODO calculate error measures and create plots----------------

    # probably mean and variance are not to be calculated so
    print(statistics.mean(errors))
    print(statistics.variance(errors))

    plt.scatter(results[:,0],results[:,1])
    plt.title('Estimation of Points')
    plt.ylabel('y/m')
    plt.xlabel('x/m')
    plt.show()

    # plot with anchor, p_true and estimated points
    
    plot_anchors_and_agent(nr_anchors, p_anchor, p_true)
    plt.scatter(results[:,0],results[:,1], c='green')
    plt.title('Estimation of Points')
    plt.axis([-8, 8, -8, 8])

    plt.show()

    Fx,x = ecdf(errors)
    plt.plot(x, Fx)
    plt.title('CDF - Estimastion of Points Error')
    plt.ylabel('Cumulative Probability')
    plt.xlabel('Squared error')
    plt.show()
    
    # plot with overlay of the contour plots over the estimated points
    mu = results.mean(axis=0)
    cov = np.cov(results.T)

    #just for defining size of plot
    if (scenario == 1) :
        ymin = -5
        ymax = -3
        xmin = 1
        xmax = 3
    elif (scenario == 2) :
        ymin = -8
        ymax = 0
        xmin = -2
        xmax = 6
    elif (scenario == 3):
        ymin = -8
        ymax = 8
        xmin = -8
        xmax = 8

    plot_gauss_contour(mu, cov, xmin,xmax,ymin,ymax)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.title('Gauss Contour of Estimated Points')
    plt.ylabel('y/m')
    plt.xlabel('x/m')
    plt.scatter(results[:,0],results[:,1],)
    plt.show()

    pass
#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    #TODO
    pass
#--------------------------------------------------------------------------------
def position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov,lambdas, p_true):
    """ estimate the position by accounting for prior knowledge that is specified by a bivariate Gaussian
    Input:
         data...distance measurements to unkown agent, nr_measurements x nr_anchors
         nr_anchors... scalar
         p_anchor... position of anchors, nr_anchors x 2
         prior_mean... mean of the prior-distribution, 2x1
         prior_cov... covariance of the prior-dist, 2x2
         lambdas... estimated parameters (scenario 3), nr_anchors x 1
         p_true... true position (needed to calculate error), 2x2 """
    # TODO
    pass
#--------------------------------------------------------------------------------
def least_squares_GN(p_anchor,p_start, measurements_n, max_iter, tol):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        measurements_n... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""
    new_point = p_start
    
    for i in range(max_iter) :

        Jf = partial(p_anchor, p_start)
        Jft = Jf.T
        diff = distanceofPoint(new_point, p_anchor, measurements_n)
        # saved for comparing for tollerance
        p_start = deepcopy(new_point)

        Jft_result = np.dot(np.dot(inv(np.dot(Jft, Jf)), Jft), diff)
        new_point[0]  -= Jft_result[0] 
        new_point[1]  -= Jft_result[1]

        if (((abs(new_point[0]) - abs(p_start[0])) <= tol) & ((abs(new_point[1]) - abs(p_start[1])) <= tol)):
            break

    return new_point
    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------

# calculate the distance between the given point and the 4 anchors and after that
# the difference between the calculated distance and the given measuraments
def distanceofPoint(point, p_anchor, measurements):
    global scenario
    global GN_mod

    final = np.zeros((4,1))
    for i in range(4) :
        x = math.sqrt((point[0] - p_anchor[i][0])**2 + (point[1] - p_anchor[i][1])**2)
        final[i] = (measurements[i] - x)

    if (scenario == 2 and GN_mod):
        return final[1:4] # only return the gaussian distributed anchors
    else:
        return final

# calculate a 4x2 Jacobi matrix for the given point
def partial(p_anchor, point):
    global scenario
    global GN_mod

    Jf = np.zeros((4,2))
    for i in range(4) :
        Jf[i][0] = - (point[0] - p_anchor[i][0]) / (math.sqrt((point[0] - p_anchor[i][0])**2 + (point[1] - p_anchor[i][1])**2))
        Jf[i][1] = - (point[1] - p_anchor[i][1]) / (math.sqrt((point[0] - p_anchor[i][0])**2 + (point[1] - p_anchor[i][1])**2))  

    if (scenario == 2 and GN_mod):
        return Jf[1:4][:] # only return the gaussian distributed anchors
    else:
        return Jf


def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax):
    
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""
    
	#npts = 100
    delta = 0.025
    X, Y = np.mgrid[xmin:xmax:delta, ymin:ymax:delta]
    pos = np.dstack((X, Y))
                    
    Z = stats.multivariate_normal(mu, cov)
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    plt.gca().set_aspect("equal")
    CS = plt.contour(X, Y, Z.pdf(pos),3,colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    #plt.show()
    return

#--------------------------------------------------------------------------------
def ecdf(realizations):   
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)
    
    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    x = np.sort(realizations)
    Fx = np.linspace(0,1,len(realizations))
    return Fx,x

#--------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file =  'reference_' + str(scenario) + '.data'
    
    data = np.loadtxt(data_file,skiprows = 0)
    reference = np.loadtxt(ref_file,skiprows = 0)
    
    return (data,reference)
#--------------------------------------------------------------------------------
def plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None):
    """ plots all anchors and agents
    Input:
        nr_anchors...scalar
        p_anchor...positions of anchors, nr_anchors x 2
        p_true... true position of the agent, 2x1
        p_ref(optional)... position for reference_measurements, 2x1"""
    # plot anchors and true position
    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_anchors):
        plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
        plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    #plt.show()
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
