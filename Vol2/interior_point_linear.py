# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Name> David Camacho
<Class> ACME
<Date> 3/30/23
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """

    # Create F and find dimensions needed
    
    F = lambda x, l, mu: np.hstack((A.T @ l + mu - c, A @ x - b, np.diag(mu) @ x))
    m, n = A.shape

    
    def _search_direction(x, l, mu, s=0.1):
        
        # Create DF and find v, b
        DF = np.block([[np.zeros((n,n)), A.T, np.eye(n)], [A, np.zeros((m,m)), np.zeros((m,n))], [np.diag(mu), np.zeros((n,m)), np.diag(x)]])
        v = np.dot(x, mu) / n       
        b = -F(x, l, mu) + np.hstack((np.zeros(m+n), v * s * np.ones(n)))
        return la.lu_solve(la.lu_factor(DF), b) 


    def _step_length(x,l, mu):

        # Find search direction
        d = _search_direction(x, l, mu)                

        # Make masks
        d_mu, d_x = d[-n:], d[:n]                       
        mu_mask, x_mask = d_mu < 0, d_x < 0            

        # Find needed alpha and delta values
        alpha_max = np.min(-mu[mu_mask] / d_mu[mu_mask]) if (d_mu < 0).any() else 1      
        d_max = np.min(-x[x_mask] / d_x[x_mask]) if (d_x < 0).any() else 1                 
        alpha_max = np.min((-mu/d_mu)[d_mu < 0])                                        
        d_max = np.min((-x/d_x)[d_x < 0])    

        # Return step length                                           
        return min(1, 0.95 * alpha_max), min(1, 0.95 * d_max), d 
    
    def solve():

        # Get dimensions of A; starting point 
        m,n = A.shape
        x, l, mu = starting_point(A,b,c)

        for _ in range(niter):

            # Get alpha, delta, and d
            alpha, delta, solution = _step_length(x, l, mu)

            # Update values
            x = x + delta * solution[:n]
            l = l + alpha * solution[n:-n]
            mu = mu + alpha * solution[-n:]

            # Check for convergence
            if np.dot(x, mu)/n < tol:
                break

        # Return optimal point and optimal value
        return x, c.T @ x
        
    return solve()

def randomLP(j, k):

    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


def test():
    j, k = 7,5
    A,b,c,x = randomLP(j,k)
    point, value = interiorPoint(A,b,c)
    print(np.allclose(x, point[:k]))


   


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    
    # Load data
    data = np.loadtxt(filename)
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m +  2*(n+1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]


    # Generate the A matrix
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    # Solve the linear program
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    # Create the Least Squares
    slope, intercept = linregress(data[:, 1], data[:, 0])[:2]
    domain = np.linspace(0,10,200)

    #  Plot data
    plt.title("Least Absolute Deviations")
    plt.scatter(data[:, 1], data[:, 0], label = 'Data', color = 'black')
    plt.plot(domain, slope * domain + intercept, label = 'Least Squares', color = 'red')
    plt.plot(domain, beta[0] * domain + b, label = "Least Absolute Deviations", color = 'blue')
    plt.legend()
    plt.tight_layout()
    plt.show()







