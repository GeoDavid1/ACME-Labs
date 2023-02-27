# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name> David Camacho        
<Class> ACME    
<Date> 10/10/22
"""

from scipy import linalg
import numpy as np

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    
    # Store the dimensions of A; make a copy of A
    m, n = np.shape(A)
    Q = A.copy()

    # Make an n x n array of all zeros

    R = np.zeros((n,n))

    

    for i in range (0, n):

        R[i,i] = linalg.norm(Q[:, i])

        # Normalize the ith column of Q

        Q[:, i] = Q[:, i]/R[i,i]
        
        for j in range(i +1, n):

            R[i,j] = np.dot(np.transpose(Q[:, j]), Q[:, i])

        # Orthogonalize the jth column of Q

            Q[:,j] = Q[:,j] - R[i,j]*Q[:, i]



    return Q, R









# Problem 2
def abs_det(A):

    # Find QR Decomposition

    Q, R = qr_gram_schmidt(A)

    # Find determinant of R

    Product_Diagonals = np.prod(np.diag(R))

    # Return desired product

    return Product_Diagonals

    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """



# Problem 3
def solve(A, b):

    # Compute Q and R

    Q, R = qr_gram_schmidt(A)

    # Find the shape of Q and R

    NumrowsR, NumcolR = np.shape(R)

    # Calculate y; initialize x

    y = np.dot(np.transpose(Q), b)

    x = np.zeros(NumrowsR)

    # Perform back subsitution to solve Rx = y for x

    for i in range(NumrowsR - 1, -1, -1):

        x[i] = y[i]/R[i,i]

        for j in range(NumrowsR - 1, i, -1):

            x[i] -= (x[j]*(R[i,j]))/(R[i,i])


    # Return x

    return x





   




    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    

    # Find shape of A; make copy of A

    m,n = np.shape(A)

    R = np.copy(A)

    # Create m x m identity matrix

    Q = np.identity(m)

    # Create sign function

    sign = lambda x: 1 if x >= 0 else -1

    for k in range(0, n):

        u = np.copy(R[k:, k])

        # u[0] is the first entry of u

        u[0] = u[0] + sign(u[0])*(linalg.norm(u))

        # Normalize u

        u = u/(linalg.norm(u))

        # Apply the reflection to R and Q

        R[k:, k:] = R[k:, k:] - 2* np.outer(u, np.dot(np.transpose(u), R[k:, k:]))
        Q[k:, :] = Q[k:, :] - 2* np.outer(u, np.dot(np.transpose(u), Q[k:, :]))



    # Return requested values

    return np.transpose(Q), R


# Problem 5
def hessenberg(A):

    # Find shape of A; make copy of A

    m,n = np.shape(A)

    H = np.copy(A)

    # Create m x m identity matrix

    Q = np.identity(m)

    # Create sign function 

    sign = lambda x: 1 if x >= 0 else -1

    for k in range(0, n-2):

        u = np.copy(H[k+1:, k])
        u[0] = u[0] + sign(u[0])*(linalg.norm(u))

        # Normalize u

        u = u/(linalg.norm(u))

        # Apply Qk to H

        H[k+1:, k:] = H[k+1:, k:] - 2*np.outer(u, np.dot(np.transpose(u), H[k+1:, k:]))

        # Apply QkT to H

        H[:, k+1:] = H[:, k+1:] - 2*np.outer(np.dot(H[:, k+1:], u), np.transpose(u))

        # Apply Qk to Q

        Q[k+1:, :] = Q[k+1:, :] - 2*np.outer(u, np.dot(np.transpose(u), Q[k+1:, :]))



    # Return requested values
    
    return H, np.transpose(Q)


    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
   