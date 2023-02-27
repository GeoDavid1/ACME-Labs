# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name>
<Class>
<Date>
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    
    sing_vals = la.svdvals(A)

    if sing_vals[-1] == 0:
        return np.inf

    return sing_vals[0]/sing_vals[-1]
   

# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """

    # Initializes the roots and arrays for storage
    w_roots = np.arange(1,21)
    k_abs = []
    k_rel = []
    real_vals = np.array([])
    imag_vals = np.array([])

    # Get the exact Wilkinson polynomial coefficients using Symy

    x,i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i,(i,1,20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    for i in range(100):
        # shift the wilkinson polynomial coefficients and calculate the new roots
        shift = np.random.normal(loc=1, scale=1e-10, size=21)
        new_coeffs = w_coeffs * shift
        new_roots = np.roots(np.poly1d(new_coeffs))

        #sort the roots and store their coordinates in the imaginary plane for plotting
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)
        real_vals = np.append(real_vals, np.real(new_roots))
        imag_vals = np.append(imag_vals, np.imag(new_roots))

        #calculate the relative and absolute condition number
        k_abs.append(la.norm(new_roots - w_roots, np.inf)/ la.norm(shift, np.inf))
        k_rel.append(k_abs[i] * la.norm(w_coeffs, np.inf)/ la.norm(w_roots, np.inf))

    #Plot the roots and return the average condition numbers
    plt.title("Perturbed Roots in the imaginary plane")
    plt.scatter(np.arange(1,21), np.imag(np.arange(1,21)), s=50, c = 'b', label = 'origin')
    plt.scatter(real_vals, imag_vals, c = 'k', marker = ',', s=1, label = "perturbed")
    plt.legend(loc = "upper left")
    plt.show()
    return np.mean(k_abs), np.mean(k_rel)


    #Find Wilkinson polynomial 
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i,1,20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    # Create 
    r = np.zeros(21)
    for i in range (0,100,1):
        for j in range (0,21,1):
            r[j] = np.random.normal(1,1e-10)

        print(r[j])


        new_coeffs = w_coeffs * r

        new_roots = np.roots(np.poly1d(new_coeffs))

        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)

        plt.scatter(np.array(w_roots.real), np.array(w_roots.imag), marker = "o", color = 'b')
        plt.scatter(np.array(new_roots.real), np.array(new_roots.imag), color = 'k', marker = ',', s = 1)

        k = la.norm(new_roots - w_roots, np.inf)/ la.norm(r,np.inf)
        abs_numbers[i] = k

        l = k * la.norm(w_coeffs, np.inf)/ la.norm(w_roots, np.inf)
        rels_numbers[i] = l

    plt.legend(['Original', 'Perturbed'])
    plt.title("Plots of Original Roots vs. Perturbed Roots")
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.show()

    



       

      


    

    



    

def examp2():

    # The roots of w are 1,2, ..., 20
    w_roots = np.arange(1,21)

    # Get the exact Wilkinson polynomial coefficients using SymPy
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    #print(w_coeffs[:6])

    # Perturb one of the coefficients very slightly
    h = np.zeros(21)
    h[1] = 1e-7
    new_coeffs = w_coeffs + h
    #print(new_coeffs[:6])

    # Use NumPy to compute the roots of the perturbed polynomial
    new_roots = np.roots(np.poly1d(new_coeffs))

    # Sort the roots to ensure that they are in the same order
    w_roots = np.sort(w_roots)
    new_roots = np.sort(new_roots)

    # Estimate the absolute condition number in the infinity norm
    k = la.norm(new_roots - w_roots, np.inf) / la.norm(h, np.inf)
    print(k)

    # Estimate the relative condition number in the infinity norm
    l  = k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)
    print(l)



# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """

    # Find the real and imaginary parts drawn from normal distributions centered at 0
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    # Compute A + H
    perturbation = A + H

    # Find eigenvalues
    A_eigenvals = la.eigvals(A)
    perturb_eigenvals = la.eigvals(perturbation)

    # Reorder eigenvalues
    perturb_eigenvals = reorder_eigvals(A_eigenvals, perturb_eigenvals)

    # Find absolute and relative condition numbers
    abs_cond = la.norm(perturb_eigenvals - A_eigenvals, 2)/la.norm(H,2)
    rel_cond = abs_cond * (la.norm(A, 2)/la.norm(A_eigenvals,2))

    # Return condition numbers
    return float(abs_cond), float(rel_cond)

def test3():

    A= np.array([[2,3],[4,5]])
    print(eig_cond(A))
    


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """

    # Create function; create bounds and meshgrid
    A = lambda x,y: np.array([[1,x],[y,1]])
    x = np.linspace(domain[0], domain[1], res)
    y = np.linspace(domain[2], domain[3], res)
    X,Y = np.meshgrid(x,y)
    k_vals = []

    # Find condition numbers of the eigenvalues for different values of x an y
    for i in range(res):
        temp_vals = []
        for j in range(res):
            temp_vals.append(np.real(eig_cond(A(X[i,j], Y[i,j]))[1]))
        k_vals.append(temp_vals)

    #Plot the relative condition nums on the meshgrid
    plt.pcolormesh(X,Y,np.array(k_vals), cmap = 'gray_r')
    plt.title("Relative Condition Numbers for A(x,y)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()    



# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    
    # Load data
    data_x, data_y = np.load("stability_data.npy").T
    A = np.vander(data_x, n+1)

    # Solve using la.inv() and la.qr()
    x_inv = la.inv(A.T @ A) @ A.T @ data_y
    Q,R = la.qr(A, mode = "economic")
    x_qr = la.solve_triangular(R, Q.T @ data_y)

    #Plot the resulting polynomials formed from b
    domain = np.linspace(0,1,1000)
    plt.plot(domain, np.polyval(x_inv, domain), label = "x = (A.TA)^-1 @ A.Tb")
    plt.plot(domain, np.polyval(x_qr, domain), label = "Rx = Q.Tb")
    plt.ylim((0,4))
    plt.scatter(data_x, data_y, color = "purple", label = "original data")
    plt.title("Best Fit for stability_data.npy")
    plt.legend(loc = "upper right")
    plt.show()

    #Return the error of each method
    err_inv = la.norm(A@x_inv - data_y)
    err_qr = la.norm(A@x_qr - data_y)
    return err_inv, err_qr

# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    err = []

    #Test different values of n
    for n in range(5,55,5):
        #Compute two different methods for the integral and store the error
        x = sy.symbols('x')
        I = x**n * sy.exp(x-1)
        integral = float(sy.integrate(I, (x,0,1)))

        # Calculate I(n) with the subfactorial of n
        I2 = float((-1)**n * (sy.subfactorial(n) - sy.factorial(n)/np.e))
        err.append(abs(integral - I2))


    # Plot the Error of each n value
    domain = np.linspace(5,50,10)
    plt.plot(domain, err)
    plt.yscale('log')
    plt.title("Error Values")
    plt.tight_layout()
    plt.show()





def examp6():

    from math import sqrt

    a = 10**20 + 1
    b = 10**20
    print(sqrt(a) - sqrt(b))
    print((a-b)/(sqrt(a) + sqrt(b)))

