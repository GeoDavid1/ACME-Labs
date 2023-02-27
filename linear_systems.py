# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""

from multiprocessing.resource_sharer import stop
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse



# Problem 1
def ref(A):

    # For each row, reduce the rows below so that there are zeros below the diagonal
    for i in range(0, len(A)):

        for j in range(i + 1, len(A)):

            A[j, i:] -= (A[j, i])/(A[i, i])*A[i,i:]


    # Return the REF matrix
    return A


# Problem 2
def lu(A):

    #Store the dimensions of A
    m,n = np.shape(A)

    # Make a copy of A
    U = np.copy(A)

    # Make an m x m identity matrix
    L = np.identity(m)

    # Perform the LU Decomposition procedure
    for j in range(0, n):
        for i in range(j+1, m):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] = U[i,j:] - L[i,j]*U[j,j:]

    # Return the Lower Triangular Matrix (L) and Upper Triangular Matrix (U)
    return L, U




# Problem 3
def solve(A, b):

   # Compute L and U
    L, U = lu(A)

    # Find length B
    Sizeb = len(b)

    # Initialize Y Values List
    YValues = []

    # Compute the vector Y with the following algorithm:
    for i in range(0, Sizeb):
        tempY = b[i]
        for j in range(0, i):
            tempY -= (L[i, j])*(YValues[j])

        YValues.append(tempY)

    # Find size of Y
    SizeY = len(YValues)

    # Initialize X Values List and set it to a list of 0's
    XValues = []

    for counter in range (0, SizeY):
        XValues.append(0)
        

    # Solve for X and append all the values of the vector X to XValues
    for i in range(SizeY - 1, -1, -1):

        tempX = (1/(U[i,i]))*(YValues[i])
        for j in range(SizeY - 1, i, -1):

            tempX -= (U[i, j])/(U[i,i])*(XValues[j])

        XValues[i] = tempX


    # Cast the list as an array
    XValuesArray = np.array(XValues)

    # Return array
    return XValuesArray

# Problem 4
def prob4():

    from scipy import linalg
    import time

    # Initialize Time Lists
    InverseTimes = []
    SolveTimes = []
    ThreeTimes = []
    FourTimes = []

    # Initialize Values of N
    NValues = [1,4,9,16,25,36,49,64,81,100, 11**2, 12**2, 13**2, 14**2, 15**2, 16**2, 17**2, 18**2, 19**2, 20**2, 21**2, 22**2, 23**2, 24**2, 25**2, 26**2, 27**2, 28**2, 29**2, 30**2, 31**2, 32**2, 33**2, 34**2, 35**2, 36**2, 37**2, 38**2, 39**2, 40**2, 41**2, 42**2]
    for nValue in NValues:

        # Initialize A and b
        A = np.random.random((nValue, nValue))
        b = np.random.random(nValue)

        # Times for 1. Invert A with la.inv() and left-multiply the inverse to b
        starttime = time.time()
        A_Inverse = linalg.inv(A)
        x = np.dot(A_Inverse, b)
        stoptime = time.time()
        timeTaken = stoptime - starttime
        InverseTimes.append(timeTaken)

        # Times for 2. Use la.solve()
        starttime = time.time()
        x = linalg.solve(A, b)
        stoptime = time.time()
        timeTaken = stoptime - starttime
        SolveTimes.append(timeTaken)

        # Times for 3. Use la.lu_factor() and la.lu_solve()
        starttime = time.time()
        L, P = linalg.lu_factor(A)
        x = linalg.lu_solve((L,P), b)
        stoptime = time.time()
        timeTaken = stoptime - starttime
        ThreeTimes.append(timeTaken)

        # Times for 4. Only la.lu_solve()
        L, P = linalg.lu_factor(A)
        starttime = time.time()
        x = linalg.lu_solve((L,P), b)
        stoptime = time.time()
        timeTaken = stoptime - starttime
        FourTimes.append(timeTaken)

    # Plot the system size n versus the execution times. 
    plt.plot(NValues, InverseTimes, 'k-', label = "la.inv()")
    plt.plot(NValues, SolveTimes, 'b-', label =  "la.solve()")
    plt.plot(NValues, ThreeTimes, 'r-', label =  "lu.factor() + lu.solve()")
    plt.plot(NValues, FourTimes, 'g-', label = "lu.solve() only")
    plt.title("Execution Times")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time (Seconds)")
    plt.legend(loc = "upper left")

    # Show plot
    plt.show()


# Problem 5
def prob5(n):

    # Set the diagonals and offsets for B
    Bdiagonals = [[1], [-4], [1]]
    Boffsets = [-1, 0, 1]

    # Create B
    B = sparse.diags(Bdiagonals, Boffsets, shape = (n, n))

    # Create A by making a block matrix with B and diagonals for the I
    A = sparse.block_diag([B]*n)
    A.setdiag(1, n)
    A.setdiag(1, -n)

    # Return A as a sparse matrix
    return A

# Problem 6

# Import needed modules

from scipy.sparse import linalg as spla
from scipy import linalg as la
import time


def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """

    # Initialize lists
    NValues = []
    convertAtoCSRTimes = []
    convertAtoNumPyTimes = []

    # Create N Values
    for i in range(2,61):

        NValues.append(i)

    # Run this loop for all n in N

    for n in NValues:

        # Generate a random matrix A and random vector b
        A = prob5(n)
        b = np.random.random(n**2)

        # Cast A to CSR format
        Acsr = A.tocsr()

        # Calculate time to solve the system in CSR format
        starttime = time.time()
        x = spla.spsolve(Acsr, b)
        stoptime = time.time()
        timeTaken = stoptime - starttime

        # Append time to list of times
        convertAtoCSRTimes.append(timeTaken)

        # Convert A to a NumPy array
        numpyA = A.toarray()

        # Calculate time to solve the system in NumPy format
        starttime = time.time()
        y = la.solve(numpyA, b)
        stoptime = time.time()
        timeTaken = stoptime - starttime

        # Append time to list of times
        convertAtoNumPyTimes.append(timeTaken)


    # Plot the system size (nxn) versus the execution times
    plt.plot(NValues, convertAtoCSRTimes, 'b', label = "Convert A to CSR")
    plt.plot(NValues, convertAtoNumPyTimes, 'k', label = "Convert A to NumPy")
    plt.xlabel("n")
    plt.ylabel("Time (Seconds)")
    plt.title("Size of Matrix A vs. Execution Times")
    plt.legend()


    # Print out the plot
    plt.show()







