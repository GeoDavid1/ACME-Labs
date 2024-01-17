# newtons_method.py
"""Volume 1: Newton's Method.
<Name>
<Class>
<Date>
"""

import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from scipy import linalg as la
from autograd import jacobian


# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """

    # Check to see if n > 1 or not
    if np.isscalar(f(x0)) == False:
        bool = False
        # If so, iterate at most maxiter times to compute zero
        for p in range(maxiter):
            y = la.solve(Df(x0), f(x0))
            x1 = x0 - alpha*y
            if la.norm(x1 - x0) < tol:
                bool = True
                break
            x0 = x1

    
        # Return approximation for zero, bool, and number of iterations computed
        return x1, bool, p+1

    

    else:
        bool = False
        # Define Newton's method function
        F = lambda x: x - alpha *f(x)/(Df(x))

        # Iterate at most maxiter times
        for k in range(maxiter):
            x1 = F(x0)              # Compute the next iteration
            if abs(x1 - x0) < tol: 
                bool = True         # Check for convergence
                break               # Upon convergence, stop iterating
            x0 = x1                 # Otherwise, continue iterating



        # Return approximation for zero, bool, and number of iterations computed
        return x1, bool, k+1

def test_prob_1():

    g = lambda x: np.e**x -2
    dg = lambda x: np.e**x

    print(newton(g, 1, dg))

def test_prob_3():

    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    df = lambda x: np.power(np.abs(x), -2/3)/3
    print(newton(f, 0.1, df, alpha = 0.32))

def test_5():

    f = lambda x: np.array([x[0] + x[1], x[0] - x[1]])
    Df = lambda x: np.array([[1, 1], [1, -1]])
    x0 = np.array([1,1])

    return newton(f, x0, Df)



   






# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    
    # Define the function to set to 0
    r = sy.symbols('r')
    F = P1 * ((1+r)**(N1) -1) - P2 * (1 - (1+r)**(-1*N2))

    # Differentiate the function
    dF = sy.diff(F, r)

    # Lambidfy the function and its derivative
    newF = sy.lambdify(r, F)
    newdF = sy.lambdify(r, dF)
    r0 = 0.1

    # Define Newton's Method Function
    funct = lambda x: x - newF(x)/(newdF(x))

    # Run a while loop to compute zero
    while np.allclose(0, newF(r0)) == False:
        r1 = funct(r0)
        r0 = r1

    # Return 0
    return float(r1)




   





# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    # Initialize empty list of number of iterations; define domain
    num_iters = []
    alpha_linspace = np.linspace(0.01,1,500)

    # Create list of number of iterations based off alpha value
    for alph in alpha_linspace:
        iters = newton(f,x0,Df, alpha = alph)[2]
        num_iters.append(iters)

    # Find minimum value and its index
    min_iters = min(num_iters)
    index = num_iters.index(min_iters)

    # Plot Number of Iterations vs. Alpha Value
    plt.plot(alpha_linspace, num_iters, 'b-')
    plt.xlabel('Alpha Value')
    plt.ylabel('Number of Iterations')
    plt.title('Number of Iterations vs. Alpha Value')
    plt.show()

    # Return Optimal Value of Alpha
    return alpha_linspace[index]

    

    



def test_prob_4():
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    df = lambda x: np.sign(x) * (1/3) * np.power(np.abs(x), -2./3)
    
    print(optimal_alpha(f, 0.01, df))




# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    
    # Define the function and ts derivative
    func = lambda x: np.array([5*x[0]*x[1] - x[0]*(1+ x[1]), -x[0]*x[1] + (1 - x[1])*(1 + x[1])])
    dfunc = lambda x: np.array([[4*x[1] - 1, 4*x[0]], [-1*x[1], -1*x[0] - 2*x[1]]])
    
    
    # Define domain
    x_lin = np.linspace(-0.25, 0, 50)
    y_lin = np.linspace(0, 0.25, 50)

    # Search throughout the rectangle domain
    for q in x_lin:
        for r in y_lin:
            x_value = np.array([q,r])

            # Compute Newton Method approximations
            new_1 = newton(func, x_value, dfunc, alpha = 1.)[0]

            # Initialize comparison arrays
            array_1 = np.array([0,1])
            array_2 = np.array([0,-1])
            array_3 = np.array([3.75, 0.25])

            # Once a valid x_value is found, return it (stop searching)
            if np.allclose(new_1, array_1) or np.allclose(new_1, array_2):
                new_2 = newton(func, x_value, dfunc, alpha = 0.55)[0]
                if np.allclose(new_2, array_3):

                    return x_value
            
            

           
        





# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """

    # Find window limits and grid domain for the plot
    rmin = domain[0]
    rmax = domain[1]
    imin = domain[0]
    imax = domain[1]

    # Construct the meshgrid
    x_real = np.linspace(rmin, rmax, res)
    x_imag = np.linspace(imin, imax, res)
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag

    # Make a copy
    Y = X_0.copy()

    # Run Newton's method
    for m in range(iters):
        X_K = -1*f(X_0)/Df(X_0) + X_0
        X_0 = X_K

    # Make Y
    for n in range(res):
        for m in range(res):
            Y[n,m] = np.argmin([(abs(X_K[n,m] - k)) for k in zeros])

    # Cast Y as type
    Y = Y.astype(int)

    # Plot Visualization Basins
    plt.pcolormesh(X_real, X_imag, Y, cmap = "brg")
    plt.title("Roots in Complex Plane", fontsize = 20)
    plt.show()
    






    


def test7():

    f = lambda x: x**3 -1
    Df = lambda x: 3*x**2
    zeros = np.array([1, -1/2 + (np.sqrt(3)/2)*1j , -1/2 - (np.sqrt(3)/2)*1j ])
    domain = [-1.5, 1.5, -1.5, 1.5]
    plot_basins(f, Df, zeros, domain, res = 1000, iters = 15)










def pract7():

    x_real = np.linspace(-1.5, 1.5, 500)
    x_imag = np.linspace(-1.5, 1.5, 500)
    X_real, X_image = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_image

    
