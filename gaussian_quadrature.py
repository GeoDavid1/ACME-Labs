# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
<Name>
<Class>
<Date>
"""

from scipy import sparse
import numpy as np
from scipy import linalg as la
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.integrate import quad

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        
        # Raise Value Error if the label is not correct
        if polytype != 'legendre' and polytype != 'chebyshev':
            raise ValueError('Polytype is not correct type')

        # Save polytype as attribute
        self.polytype = polytype

        # Find weights of Legendre polynomials
        if self.polytype == 'legendre':
            w = lambda x: 1
        
        # Find weights of Chebyshev polynomials
        else:
            w = lambda x: 1/ np.sqrt(1-x**2)

        # Save lambda function as an attribute
        self.w = w
        self.inverse = lambda x: 1/self.w(x)
        self.points, self.weights = self.points_weights(n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """

        # Legendre coefficient calculator
        def leg_coeff(k):
            return k**2/(4*k**2 -1)

        # Chebyshev coefficient calculator
        def cheb_coeff(k):
            if k == 1:
                return 1/2
            else:
                return 1/4
    
        # Compute the eigenvalues and eigenvectors using Legendre polynomials
        if self.polytype == 'legendre':

            # Create the Jacobian Matrix
            diags = []
            for j in range(1, n):
                diags.append(np.sqrt(leg_coeff(j)))
            diags = [diags, diags]
            offsets = [-1, 1]
            JacobMatr = sparse.diags(diags, offsets, shape = (n,n))

            # Evaluate eigenvalues and eigenvectors; Find weights
            val, vect = la.eig(JacobMatr.toarray())
            vec0 = vect[0,:]
            weights = 2*vec0**2
            

        # Compute the eigenvalues and eigenvectors using Chebyshev polynomials 
        else:

            # Create the Jacobian Matrix
            diags = []
            for k in range(1,n):
                diags.append(np.sqrt(cheb_coeff(k)))
            diags = [diags, diags]
            offsets = [-1,1]
            JacobMatr = sparse.diags(diags, offsets, shape = (n,n))

            # Evaluate eigenvalues and eigenvectors; Find weights
            val, vect = la.eig(JacobMatr.toarray())
            vec0 = vect[0,:]
            weights = np.pi*vec0**2
    
        # Return points and weights
        return val, weights


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        
        # Calculate g(x)
        g = f(self.points) * self.inverse(self.points)
        
        # Find the inner product and return it
        return np.dot(self.weights.T, g)


    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        
        # Define h(x) 
        h = lambda x: f(((b-a)/2)*x + (a+b)/2)

        # Calculate g(x) with h(x)
        g = h(self.points) * self.inverse(self.points)

        # Return the approximation of the integral
        return ((b-a)/2) * (np.dot(self.weights.T, g))

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """

        # Define h(x)
        h = lambda x, y: f((b1 - a1) * x/2 + (a1 + b1)/2, (b2 - a2) * y/2 + (a2 + b2)/2)

        # Calculate g(x,y) with h(x,y)
        g = []
        for zi in self.points:
            for zj in self.points:
                g.append(h(zi, zj) * self.inverse(zi) * self.inverse(zj))

        # Find out the double summation
        Sum = 0
        k = 0
        for i in range(0, len(self.weights)):
            for j in range(0, len(self.weights)):

                Sum += self.weights[i] * self.weights[j] * g[k]
                k += 1
        
        # Return final product
        return (b1 - a1)*(b2 - a2) * Sum/4 





       
    


        


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    # Initialize Error Arrays
    Errors_lagrange = []
    Errors_cheb = []
    N = []

    # Compute "Exact" Value; define f(x)
    exact_val = norm.cdf(2) - norm.cdf(-3)
    f = lambda x: (1/(np.sqrt(2*np.pi))*(np.exp(-x**2/2)))
   
    # Compute the Lagrange and Chebyshev errors
    for n in range(5, 55, 5):
        N.append(n)
        GQ1_leg = GaussianQuadrature(n, polytype = 'legendre')
        GQ2_cheb = GaussianQuadrature(n, polytype = 'chebyshev')
        lagrange_val = GQ1_leg.integrate(f, -3, 2)
        cheb_val = GQ2_cheb.integrate(f, -3, 2)
        Errors_lagrange.append(np.abs(lagrange_val - exact_val))
        Errors_cheb.append(np.abs(cheb_val - exact_val))

    # Compute Scipy Error
    Quad = quad(f, -3, 2)[0]
    Error_Quad = np.abs(exact_val - Quad)
    errors_quad = []
    for m in range(0,10):
        errors_quad.append(Error_Quad)


    # Plot Error Approximations
    plt.semilogy(N, Errors_lagrange, 'b.-', base = 2, lw = 2, label = "Legendre Error")
    plt.semilogy(N, Errors_cheb, 'g.-', base = 2, lw = 2, label = "Chebyshev Error")
    plt.semilogy(N, errors_quad, 'k.-', base = 2, lw = 2, label = "Error of Scipy.Integrate.Quad()")
    plt.title("Errors of Approximation")
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    




def test3():
    GQ = GaussianQuadrature(100, 'legendre')
    #print(GQ.points_weights(5))
    f = lambda x: 1 / np.sqrt(1 - x**2)
    print(GQ.basic(f))

def test6():

    f = lambda x, y: np.sin(x) + np.cos(y)
    GQ6 = GaussianQuadrature(100, 'legendre')
    print(GQ6.integrate2d(f, -10, 10, -1, 1))
