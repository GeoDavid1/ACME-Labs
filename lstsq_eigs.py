# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """

    # Find QR decomposition
    Q, R = la.qr(A, mode = "economic")

    # Solve normal equation
    ProductQTransposeB = np.dot(np.transpose(Q), b)
    x = la.solve_triangular(R, ProductQTransposeB)

    # Return desired value
    return x


# Problem 2
def line_fit():

    # Load data; find dimension of data
    data = np.load("housing.npy")
    DimofData = len(data)

    # Create a vector of ones
    Ones = np.ones(DimofData)

    # Initialize needed lists
    Years = []
    PriceIndices = []

    # Add data to Years and PriceIndices
    for datapoint in data:
        Years.append(datapoint[0])
        PriceIndices.append(datapoint[1])

    # Create the matrix to solve the least squares solution
    YearsArray = np.array(Years)
    YearsExtraArray = np.column_stack((YearsArray, Ones))
        
    # Find the least squares solution
    LeastSquaresSolution = least_squares(YearsExtraArray, PriceIndices)

    # Find slope and YIntercept
    Slope = LeastSquaresSolution[0]
    YIntercept = LeastSquaresSolution[1]

    # Define domain
    x = np.linspace(0, 16, 200)

    # Plot data
    plt.plot(Years, PriceIndices, '*', label = "Data Points")
    plt.plot(x, Slope*x + YIntercept, 'b', label = "Least Squares Fit")
    plt.title("Year vs. Purchase-Only Housing Price Index")
    plt.xlabel("Year(0 = 2000)")
    plt.ylabel("Purchase-Only Housing Price Index")
    plt.legend()
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """

    # Load Data
    data = np.load("housing.npy")

    # Initialize Lists
    Years = []
    DegreeThreeYears = []
    DegreeSixYears = []
    DegreeNineYears = []
    DegreeTwelveYears = []
    PriceIndices = []

    for datapoint in data:

        # Initialize data lists
        ThreeNeededData = []
        SixNeededData = []
        NineNeededData = []
        TwelveNeededData = []

        # Find matrix for polynomial of degree 3 data
        for i in range(3, -1, -1):
            ThreeNeededData.append(datapoint[0]**i)
        DegreeThreeYears.append(ThreeNeededData)

        # Find matrix for polynomial of degree 6 data
        for j in range(6, -1, -1):
            SixNeededData.append(datapoint[0]**j)
        DegreeSixYears.append(SixNeededData)

        # Find matrix for polynomial of degree 9 data
        for k in range(9, -1, -1):
            NineNeededData.append(datapoint[0]**k)
        DegreeNineYears.append(NineNeededData)

        # Find matrix for polynomial of degree 12 data
        for l in range(12, -1, -1):
            TwelveNeededData.append(datapoint[0]**l)
        DegreeTwelveYears.append(TwelveNeededData)

    # Make Years and PriceIndices matrices
    for datapoint in data:
        Years.append(datapoint[0])
        PriceIndices.append(datapoint[1])

    # Find least squares solution constants for polynomial of degree 3,6,9,12
    ThreeX = la.lstsq(DegreeThreeYears, PriceIndices)[0]
    SixX = la.lstsq(DegreeSixYears, PriceIndices)[0]
    NineX = la.lstsq(DegreeNineYears, PriceIndices)[0]
    TwelveX = la.lstsq(DegreeTwelveYears, PriceIndices)[0]

    # Create functions with these constants
    f = np.poly1d(ThreeX)
    g = np.poly1d(SixX)
    h = np.poly1d(NineX)
    i = np.poly1d(TwelveX)

    # Define domain
    x = x = np.linspace(0, 16, 200)

    # Plot degree 3 polynomial least squares solution
    plt.subplot(221)
    plt.xlabel("Year(0 = 2000)")
    plt.ylabel("Purchase-Only Housing Price Index")
    plt.plot(Years, PriceIndices, '*', label = "Data Points")
    plt.plot(x, f(x), 'b', label = '3rd-Degree')
    plt.title("3rd Degree Polynomial Best Fit")
    plt.legend()

    # Plot degree 6 polynomial least squares solution
    plt.subplot(222)
    plt.xlabel("Year(0 = 2000)")
    plt.ylabel("Purchase-Only Housing Price Index")
    plt.plot(Years, PriceIndices, '*', label = "Data Points")
    plt.plot(x, g(x), 'b', label = '6th-Degree')
    plt.title("6th Degree Polynomial Best Fit")
    plt.legend()

    # Plot degree 9 polynomial least squares solution
    plt.subplot(223)
    plt.xlabel("Year(0 = 2000)")
    plt.ylabel("Purchase-Only Housing Price Index")
    plt.plot(Years, PriceIndices, '*', label = "Data Points")
    plt.plot(x, h(x), 'b', label = '9th-Degree')
    plt.title("9th Degree Polynomial Best Fit")
    plt.legend()

    # Plot degree 12 polynomial least squares solution
    plt.subplot(224)
    plt.xlabel("Year(0 = 2000)")
    plt.ylabel("Purchase-Only Housing Price Index")
    plt.plot(Years, PriceIndices, '*', label = "Data Points")
    plt.plot(x, i(x), 'b', label = '12th-Degree')
    plt.title("12th Degree Polynomial Best Fit")
    plt.legend()
    
    # Show plots
    plt.show()
   
    

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """

    # Load ellipse; Make appropriate matrices
    xk, yk = np.load("ellipse.npy").T
    S = np.column_stack((xk**2, xk, xk*yk, yk, yk**2))
    t = np.ones(S.shape[0])

    # Find constants for least squares solution
    a,b,c,d,e = la.lstsq(S,t)[0]

    # Plot the data and the least squares ellipse solution
    plot_ellipse(a,b,c,d,e)
    plt.plot(xk, yk, 'k*')
    plt.axis("equal")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Least Squares Solution for Ellipse")
    plt.show()
    


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """

    # Find dimensions of A
    m,n = np.shape(A)

    # Make a random vector of length n; normalize that vector
    x_naught = np.random.random(n)
    x_naught = x_naught/(la.norm(x_naught))

    # For N iterations, do the following:
    for k in range(0, N):

        # Run the power method
        tempVar = x_naught
        x_naught = np.dot(A, x_naught)
        x_naught = x_naught/(la.norm(x_naught))

        # If the sequence has not converged to an eigenvalue, continue
        if np.abs(la.norm(x_naught - tempVar)) > tol:
            continue

        # If the sequence has converged to an eigenvalue, break out of the for loop
        else:
            break

    # Return desired results
    return np.dot(x_naught.T, np.dot(A, x_naught)), x_naught
    


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """

    m,n = np.shape(A)

    # Put A in upper Hessenberg form
    S = la.hessenberg(A)

    # Get the QR Decomposition of Ak
    for k in range(0, N):
        Q,R = la.qr(S)

        # Recombine Rk and Qk into Ak+1
        S = np.dot(R,Q)

    # Initialize an empty list of eigenvalues; initialize counter
    eigs = []
    i = 0

    # Run this loop for every column in A

    while i < n:
        
        # If the submatrix is 1x1, simply append the only entry to eigs
        if i == n-1 or abs(S[i+1, i]) < tol:
            eigs.append(S[i,i])

        # If the submatrix is 2x2, calculate the eigenvalues using the quadratic formula

        else:

            # Initialize constants for the quadratic formula
            a = S[i,i]
            b = S[i+1, i]
            c = S[i, i+1]
            d = S[i+1, i+1]

            # Calculate eigenvalues
            Eigenvalue1 = ((a+d) + cmath.sqrt((a+d)**2 - 4*(a*d - b*c)))/2
            Eigenvalue2 = ((a+d) - cmath.sqrt((a+d)**2 - 4*(a*d - b*c)))/2

            # Append eigenvalues
            eigs.append(Eigenvalue1)
            eigs.append(Eigenvalue2)

            # Increment counter
            i = i + 1

        # Increment counter
        i = i + 1

    # Return eigenvalues
    return eigs
    
    