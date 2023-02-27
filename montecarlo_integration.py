# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy import stats
from matplotlib import pyplot as plt

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """

    # Compute volume of n-dimensionoal cube
    tot_vol = 2**n

    # Get N random points in the n-D domain
    points = np.random.uniform(-1,1,(n, N))

    # Determine how many points are within the sphere
    lengths = la.norm(points, axis = 0)
    num_within = np.count_nonzero(lengths < 1)

    # Return estimate for the volume of the n-dimensional unit ball.
    vol = tot_vol * (num_within/ N)
    return vol
    
def test1():
    # Get 2000 random points in the 2-D domain [-1,1]x[-1x1]
    points = np.random.uniform(-1,1,(2,2000))

    # Determine how many points are within the circle
    lengths = la.norm(points, axis = 0)
    num_within = np.count_nonzero(lengths < 1)

    # Estimate the circle's area
    print(4* num_within /2000)



# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    
    # Compute omega
    omega = b-a

    # Get N random points in the domain
    points = np.random.uniform(a,b, N)

    # Find average and return approximation
    avg = sum(f(p) for p in points)/N
    return omega * avg


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    
    # Get N random points
    points = np.random.uniform(0,1,(N, len(mins)))

    # Find total volume of higher-dimensional box
    values = np.array(maxs) - np.array(mins)
    vol = np.product(values)

    #Translate all the points to their respective boundaries
    for point in points:
        for i in range(0,len(mins),1):
            point[i] = point[i] * (maxs[i] - mins[i]) + mins[i]

    # Estimate the integral d
    avg = np.sum(np.array([f(p) for p in points]))/N
    return float(vol) * avg


def test3():
    f = lambda x: 3*x[0] - 4*x[1] + x[1]**2
    print(mc_integrate(f, [1,-2], [3,1]))

    f0 = lambda x: x**2
    print(mc_integrate(f0, [-4], [2]))

    f1 = lambda x: x[0]**2 + x[1]**2
    print(mc_integrate(f1, [0,0], [1,1]))

    f2 = lambda x: 3*x[0] - 3*x[1] + x[1]**2
    print(mc_integrate(f2, [1,-2], [3,1]))

    f3 = lambda x: np.sin(x[0]) - x[1]**3
    print(mc_integrate(f3, [-1,-2], [2,1]))




# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    
    # Initialize function and bounds
    f = lambda x: (1/(2*np.pi)**(4/2)) * np.exp(-1*(x.T @ x)/2)
    mins = [-3/2, 0, 0, 0]
    maxs = [3/4, 1, 1/2, 1]
    errors = []

    # Compute the integral with Scipy
    means, cov = np.zeros(4), np.eye(4)
    exact_value = stats.mvn.mvnun(mins, maxs, means, cov)[0]


    # Get 20 integer values of N that are roughly logarithmically spaced
    N = np.logspace(1,5, 20)
    N = np.around(N)
    N = [int(n) for n in N]

    # Compute the relative error for each value of N
    for n in N:
        integral_value = mc_integrate(f, mins, maxs, n)
        error = np.abs(exact_value - integral_value)/np.abs(exact_value)
        errors.append(error)


    # Plot the relative error and 1/sqrt(N) for comparison
    plt.loglog(N, errors, 'b-', label = "Relative Error")
    plt.loglog(N, 1/np.sqrt(N), 'r', label =  "1/(np.sqrt(N))")
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.title("Relative Error vs. Number of Samples")
    plt.legend()
    plt.show()



    