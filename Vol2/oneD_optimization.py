import numpy as np
import scipy as sp
from scipy.optimize import linesearch
#from autograd import numpy as jnp
#from autograd import grad

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=100):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    
    # Set the initial minimizer approximation as the interval midpoint
    x0 = (a+b)/2                        
    fi = (1 + np.sqrt(5))/2
    count = 0

    # Iterate only maxiter times at most
    for i in range(1, maxiter+1):
        count += 1
        c = (b-a)/(fi)
        a_new = b-c
        b_new = a + c

        # Get new boundaries for the search interval
        if f(a_new) <= f(b_new):
            b = b_new
        else:
            a = a_new
        
        # Set the minimizer approximation as the interval midpoint
        x1 = (a+b)/2
        
        # Stop iterating if the approximation stops changing enough
        if np.abs(x0 - x1) < tol:
            return x1, True, count
            break
        x0 = x1
    
    # Return zero
    return x1, False, count




# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=100):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    
    # Define Newton's method to find zeros
    F = lambda x: x - df(x)/d2f(x)

    # Evaluate next iterationoo
    for k in range(maxiter):
        x1 = F(x0)
        if abs(x1 - x0) < tol:
            break
        x0 = x1

    # Determine whether Newton's method converges or not
    if np.allclose(df(x1), 0, atol = 10e-2):
        bool = True
    else:
        bool = False

    # Return desired values
    return x1, bool, k+1


def test2():

    df = lambda x: 2*x + 5*np.cos(5*x)
    d2f = lambda x: 2 - 25*np.sin(5*x)
    print(sp.optimize.newton(df, x0=0, fprime = d2f, tol = 1e-10, maxiter = 500))
    print(newton1d(df, d2f, 0))


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=10):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    

    # Initialize variables
    x_vals = [x0, x1]
    y_vals = [df(x0), df(x1)]
    conv = False
    n = 0
    while n < maxiter:
        # Calculate new x and perform tolerance check
        x_vals.append((x_vals[-2] * y_vals[-1] - x_vals[-1]*y_vals[-2])/(y_vals[-1] - y_vals[-2]))
        if np.abs(x_vals[-1] - x_vals[-2]) < tol:
            conv = True
            # Break if needed
            break
        y_vals.append(df(x_vals[-1]))
        n += 1
    return x_vals[-1], conv, n
              

def test3():

    df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    print(sp.optimize.newton(df, x0 = 0, tol = 1e-10, maxiter = 500))
    print(secant1d(df, 0, -1, tol = 1e-5, maxiter = 100))



# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
   
    # Compute these values only once
    Dfp = Df(x).T @ p          
    fx = f(x)

    # Run this loop until it satisfies the Armijo-Goldstein conditions
    while (f(x + alpha*p) > fx + c*alpha*Dfp):
        alpha = rho*alpha
    
    # Return step size
    return alpha


"""def test4():
    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
    #x = jnp.array([150., 0.03, 40.])
    #p = jnp.array([-0.5, -100, -4.5])
    phi = lambda alpha: f(x + alpha*p)
    print(backtracking(f,Df, x, p))"""


    
