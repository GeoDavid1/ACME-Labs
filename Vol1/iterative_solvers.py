# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from scipy.sparse import diags
from scipy import sparse


# Helper function
def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.
    Parameters:
    n (int): The dimension of the system.
    num_entries (int): The number of nonzero values.
    Defaults to n^(3/2)-n.
    as_sparse: If True, an equivalent sparse CSR matrix is returned.
    Returns:
    A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """


    if num_entries is None:
        num_entries = int(n**1.5) - n

    A = sparse.dok_matrix((n,n))
    rows = np.random.choice(n, size=num_entries)
    cols = np.random.choice(n, size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    B = A.tocsr() # convert to row format for the next step
    for i in range(n):
        A[i,i] = abs(B[i]).sum() + 1

    return A.tocsr() if as_sparse else A.toarray()

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot = False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """

    # Initialize counter; x_0, and D_inv
    counter = 0
    x_0 = np.zeros(len(A))
    diag = np.diag(A)
    diag_inv = 1/diag
    D_inv = np.diag(diag_inv)
    #print(D_inv)
    

    # If Plot == True, then make sure to plot the errors
    errors = []
    if plot == True:

        # Iterate for maxiter number of iterations
        while(counter < maxiter):
            x_temp = x_0 + D_inv @ (b - A@x_0)

            # Find errors
            error = la.norm(A@x_0 -b, ord = np.inf)
            errors.append(error)

            # Break if tolerance condition is met
            if la.norm(x_0 - x_temp, ord = np.inf) < tol:
                x_0 = x_temp
                break

            else:
                x_0 = x_temp

        # Plot results
        iters = np.arange(0,len(errors))
        plt.semilogy(iters, errors, 'b-')
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()


    # Else, don't plot
    else:

        # Iterate for maxiter number of iterations
        while(counter < maxiter):
            x_temp = x_0 + D_inv @ (b - A@x_0)

            # Break if tolerance condition is met
            if la.norm(x_0 - x_temp, ord = np.inf) < tol:
                x_0 = x_temp
                break

            else:
                x_0 = x_temp

    return x_0



def test1():
    A = diag_dom(3)
    b =  np.random.rand(3)
    #print(b)
    #print(A)
    print(jacobi(A,b, plot = True))





# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """

    # Initialize variables
    counter = 0
    x_0 = np.zeros(len(A))
    errors = []

    # Plot it if needed
    if plot == True:
    
        while(counter < maxiter):

            # Find errors
            error = la.norm(A@x_0 - b, ord = np.inf)
            errors.append(error)

            # Run through algorithm
            for i in range(0, len(x_0)):
                x_copy = x_0.copy()
                x_0[i] = x_0[i] + (1/A[i][i])*(b[i] - A[i].T @ x_0)

            # Check tolerance
            if la.norm(x_copy - x_0, ord = np.inf) < tol:
                break
            
            # Increment counter
            counter += 1

        
        # Plot results
        iters = np.arange(0, len(errors))
        plt.semilogy(iters, errors, 'b-')
        plt.title("Convergence of Gauss-Seidel Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()

    else:
        while(counter < maxiter):
            # Find errors
            for i in range(0, len(x_0)):
                x_copy = x_0.copy()
                x_0[i] = x_0[i] + (1/A[i][i])*(b[i] - A[i].T @ x_0)

            # Check tolerance
            if la.norm(x_copy - x_0, ord = np.inf) < tol:
                break

            # Increment counter
            counter += 1

    return x_0 

def test2():
    A = diag_dom(3)
    b =  np.random.rand(3)
    print(b)
    print(A)
    print(gauss_seidel(A,b, plot = False))
    print(jacobi(A,b))



# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """

    # Initialize counter and x_0
    counter = 0
    x_0 = np.zeros(len(b))
    
    # Create while loop
    while(counter < maxiter):
        
        # For all the elements in x, do the Gauss-Seidel Sparse calculation
        for i in range(0, len(x_0)):
            x_copy = x_0.copy()
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            Aix = A.data[rowstart:rowend] @ x_0[A.indices[rowstart:rowend]]

            x_0[i] = x_0[i] + (1/(A[i,i]))*(b[i] - Aix)

        # If we meet convergence tolerance, break
        if la.norm(x_copy - x_0, ord = np.inf) < tol:
            break

        counter += 1

    # Return needed value
    return x_0

def test3():
    A = diag_dom(4, as_sparse = True)
    b =  np.random.rand(4)
    print(b)
    print(A)
    print(gauss_seidel_sparse(A,b))



# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """

    converged = False
    # Initialize counter and x_0
    counter = 0
    x_0 = np.zeros(len(b))
    
    # Create while loop
    while(counter < maxiter):
        
        # For all the elements in x, do the Gauss-Seidel Sparse (with omega) calculation
        for i in range(0, len(x_0)):
            x_copy = x_0.copy()
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            Aix = A.data[rowstart:rowend] @ x_0[A.indices[rowstart:rowend]]

            x_0[i] = x_0[i] + (omega/(A[i,i]))*(b[i] - Aix)

        # If we meet convergence tolerance, break
        if la.norm(x_copy - x_0, ord = np.inf) < tol:
            converged = True
            break

        counter += 1

    # Return needed value
    return x_0, converged, counter + 1
    


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """

    # Create diagonal arrays
    main_diag = np.full((n**2,), -4)
    upper_diag = np.full((n**2 -1,), 1)
    lower_diag = np.full((n**2 - 1,), 1)

    # Create A and b
    A = diags([lower_diag, main_diag, upper_diag], [-1,0,1], shape = (n**2,n**2), format = 'csr')
    b = np.zeros(n**2)
    b[0::n] = -100
    b[n-1::n] = -100

    # Find values of sor
    u, bool_value, num_iter = sor(A,b, omega)

    # Reshape u
    u = np.reshape(u, (n,n))

    # Plot colormap of u
    if plot == True:
        plt.pcolormesh(u, cmap = 'coolwarm')
        plt.colorbar()
        plt.title("Heat Map")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
       
    # Return needed values
    return u, bool_value, num_iter
    



# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    
    # Find omegas and iterations for those omegas
    omegas = [1 + 0.05*n for n in range(0,20,1)]
    iterations = [hot_plate(20, omega,tol = 1e-2, maxiter = 1000)[2] for omega in omegas]

    # Plot data
    plt.plot(omegas, iterations)
    plt.title("Number of computed iterations vs. omega")
    plt.xlabel("Omega")
    plt.ylabel("Number of Iterations")
    plt.show()

    # Return the value of omega that results in the least number of iterations
    argmin = np.argmin(iterations)
    return omegas[argmin]
