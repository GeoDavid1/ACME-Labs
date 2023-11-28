# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name> David Camacho    
<Class> 
<Date> 2/21/21
"""
from scipy.optimize import minimize_scalar as ms
import numpy as np
from autograd import jacobian
from scipy import linalg as la
from scipy.optimize import rosen, rosen_der

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    
    # Initialize variables
    xk = x0
    iters = 0
    conv = False

    # Calculate the argmin of the one-dimensional optimization problem
    for i in range(maxiter):
        iters += 1
        g = lambda alph: f(xk - alph*Df(xk).T)
        alpha = ms(g).x

        # Do the gradient descent
        xk_1 = xk - alpha*Df(xk).T

        # Iterate until the norm is less than tolerance
        if la.norm(Df(xk_1), ord = np.inf) < tol:
            conv = True
            xk = xk_1
            break
        
        # Restart the process
        else:
            xk = xk_1

    # Return desired values
    return xk, conv, iters

def test1():

    f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    Df = lambda x: np.array([4*x[0]**3, 4*x[1]**3, 4*x[2]**3])
    x0 = np.array([150,200,96.5])
    print(steepest_descent(f, Df, x0))

    





# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """

    # Initialize variables
    conv = False
    n = len(Q)

    # Start Algorithm 12.1
    r0 = Q@x0 - b
    d0 = -1*r0
    k = 0

    # Rrun this loop at most n number of times
    while k < n:
        alpha_k = r0.T@r0/(d0.T@Q@d0)
        x_k1 = x0 + alpha_k*d0
        r_k1 = r0  + alpha_k*Q@d0
        beta_k1 = r_k1.T@r_k1/(r0.T@r0)
        d_k1 = -1*r_k1 + beta_k1*d0
        k = k + 1

        # Stop look if the norm of r is less than tolerance
        if la.norm(r_k1) < tol:
            conv = True
            x0 = x_k1
            r0 = r_k1
            d0 = d_k1
            break

        # If not, keep going
        else:
            x0 = x_k1
            r0 = r_k1
            d0 = d_k1

    # Return desired values
    return x0, conv, k
    



    while la.norm(r[k]) >= tol and k < n:
        alpha[k] = (r[k].T @ r[k]) / (d[k].T @ Q @ d[k])
        x[k+1] = x[k] + alpha[k]*d[k]
        r[k+1] = r[k] + alpha[k]*Q*d[k]
        beta[k+1] = (r[k+1].T @ r[k+1])/(r[k].T @ r[k])
        d[k+1] = -1*r[k+1] + beta[k+1]*d[k]
        k = k + 1
    
    return x[0]


def test2():

    Q = np.array([[2,0], [0,4]])
    b = np.array([1,8])
    x0 = np.array([0.5,22.2])

    print(conjugate_gradient(Q, b, x0))
        


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """

    # Initialize variables
    conv = False
    r0 = -df(x0).T
    d0 = r0
    g = lambda alph: f(x0 + alph*d0)

    # Find the argmin of the optimization function
    alpha0 = ms(g).x

    # Find the new value and set the stage for the while loop
    x1 = x0 + alpha0*d0
    k = 1
    r_k = r0
    d_k = d0
    x_k = x1
    alpha_k = alpha0

    # Run this loop at most maxiter number of times
    while k < maxiter:
        r_k1 = r_k
        r_k = -df(x_k).T
        beta_k = (r_k.T @ r_k)/(r_k1.T @ r_k1)
        d_k = r_k + beta_k * d_k
        h = lambda alph: f(x_k + alph * d_k)
        alpha_k = ms(h).x
        x_k = x_k + alpha_k*d_k
        k = k + 1

        # Break look if la.norm() is less than tolerance
        if la.norm(r_k) < tol:
            conv = True
            break


    # Return desired values
    return x_k, conv, k

def test3():
    print(nonlinear_conjugate_gradient(rosen,rosen_der, np.array([10,10])))





# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    
    # Load data and find y values
    data = np.loadtxt(filename)
    y_values = data[:,0]
    Y_values = y_values.copy()
    data[:,0] = 1

    # Build Q and b and find solution
    A = data
    b = Y_values
    Q = A.T @ A
    sol = A.T @ b

    # Return desired value
    return conjugate_gradient(Q, sol, x0)[0]


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        
        

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    raise NotImplementedError("Problem 6 Incomplete")
