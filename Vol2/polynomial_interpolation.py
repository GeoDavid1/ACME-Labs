# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
import scipy.linalg as la

plt.figure(figsize = (25, 8), dpi = 80)

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
   
    #Compute denominator of each Lj
    Ljdenom = [np.product([xj - xk for xk in xint if xk != xj]) for xj in xint]

    # Evaluates Lj at all points in the computational domain at each of the n Ljs using the denominator
    L = np.array([[np.product([x - xint[k] for k in np.arange(len(xint)) if k != j])/Ljdenom[j] for x in points] for j in np.arange(len(xint))])

    # Combines the y values of the interpolation points and the evaluated Lagrange basis functions from problem 1
    return L.T @ yint



# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """

        # Store xint and yint
        self.x = xint
        self.y = yint

        #Calculates barycentric weights and stores the resulting array as a class attribute
        self.b = [1 / np.product([xj-xk for xk in xint if xk != xj]) for xj in xint]



    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """

        # Return evaluated polynomial
        return [self.y[np.where(self.x == x0)[0][0]] if x0 in self.x else sum([self.y[i] * self.b[i] / (x0 - self.x[i] + .00000001) for i in np.arange(len(self.x))]) / sum([self.b[i] / (x0 - self.x[i] + .00000001) for i in np.arange(len(self.x))]) for x0 in points]

            

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        
        # Update x and y
        self.x = np.append(self.x, xint)
        self.y = np.append(self.y, yint)

        # Find new weights
        self.b = [1/ np.product([xj - xk for xk in self.x if xk != xj]) for xj in self.x]

# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """

    # Initializes n = 2^2,..., 2^8 and error lists along with domain
    n = [2**i for i in range(2,9)]
    errors = []
    cheberr = []
    x = np.linspace(-1,1,400)
    for a in n:
        # Defines Runge's function and n equally spaced points
        f = lambda x: 1/ (1 + 25*x**2)
        points = np.linspace(-1,1,a)

        # Interpolates the function at those points using SciPy's BarycentricInterpolator
        poly = BarycentricInterpolator(points, f(points))
        intx = poly(x)

        # Calculates and stores the error
        errors.append(np.linalg.norm(f(x) - intx, ord = np.inf))

        # Calculates Chebyshev extremal points and interpolates the function, and saves the error
        cheby = np.cos((np.arange(a) + 1.0) * np.pi / a)

        bary = BarycentricInterpolator(cheby, f(np.array(cheby)))
        cheby = bary(x)

        cheberr.append(np.linalg.norm(f(x) - cheby, ord = np.inf))

    # Plot the errors of each method against the number of interpolating points n in a loglog plot
    plt.loglog(n, errors, base = 2, label = 'Equally spaced point error')
    plt.loglog(n, cheberr, base = 2, label = 'Chebyshev extremal point error')
    plt.legend()
    plt.xlabel("# of points")
    plt.ylabel("Absolute Error")
    plt.title("Errors")
    plt.show()

# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    
    # Calculates Chebyshev Extremizers and evaluates f at those points
    y = np.cos((np.pi * np.arange(2*n)) / n)
    samples = f(y)

    # Computes the coefficients of the degree-n Chebyshev interpolation of f
    coeffs = np.real(np.fft.fft(samples))[:n+1]/n
    coeffs[0] /= 2
    coeffs[n] /= 2
    return coeffs


    

# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """

    # Import data
    data = np.load('airdata.npy')

    # Takes n+1 Chebyshev extrema and finds the closest match in the non-continuous data found in the variable data
    fx = lambda a,b,n: .5 * (a+b+ (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0,b, 8784)
    points = fx(a,b,n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis = 0)

    # Calculates the barycentric weights
    poly = Barycentric(domain[temp2], data[temp2])
    poly = poly(domain)

    # Plots original data and approximating polynomial on the same domain on two separate subplots
    plt.subplot(121)
    plt.plot(domain, data, 'b.')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Given Data')
    plt.subplot(122)
    plt.plot(domain, poly)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Interpolating Polynomial')
    plt.show()




