"""Volume 1: The SVD and Image Compression."""

import numpy as np
from scipy import linalg as la
import math
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import pyplot as plt

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    # Find A Hermitian
    AHermitian = A.conj().T

    # Calculate the eigenvalues and eigenvectors of AHA
    Lambda, V = la.eig(AHermitian @ A)

    # Calculate the singular values of A
    SingularValues = np.sqrt(Lambda)

    # Sort the singular values and eigenvectors from greatest to least
    SVIndices = np.flip(SingularValues.argsort())
    SingularValues = SingularValues[SVIndices]
    V = V[:, SVIndices]

    # Count the number of nonzero singular values (the rank of A)
    Count = np.sum(SingularValues > tol)

    # Keep only the positive singular values
    Sigma1 = SingularValues[:Count]

    # Keep only the corresponding eigenvectors
    V1 = V[:, :Count]

    # Construct U with array broadcasting
    U1 = (A@V1)/Sigma1

    # Return U, Sigma, and VH
    print(A, V1)
    return U1, Sigma1, V1.conj().T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """

    # Generate a 2 x 200 matrix representing a set of 200 points on the unit circle
    S = np.zeros((2,200))
    for i in range (0, 200,1):
        S[0, i] = math.cos((i/200)*2*math.pi)
        S[1, i] = math.sin((i/200)*2*math.pi)

    # Make the elementary matrix E
    E = np.array([[1,0,0], [0,0,1]])

    # Find the SVD decomposition
    U, Sigma, Vh = la.svd(A)

    # Plot S
    ax1 = plt.subplot(221)
    Cos = S[0, :]
    Sin = S[1, :]
    ax1.plot(Cos, Sin, 'b-')
    ax1.plot(E[0], E[1], 'g-')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax1.set_title("S")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis("equal")

    # Plot VHS
    ax2 = plt.subplot(222)
    VhS = Vh @ S
    VhE = Vh @ E
    ax2.plot(VhS[0], VhS[1], 'b-')
    ax2.plot(VhE[0], VhE[1], 'g-')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax2.set_title("VHS")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis("equal")

    # Plot SigmaVHS
    ax3 = plt.subplot(223)
    SigmaVHS = np.diag(Sigma) @ VhS
    SigmaVHE = np.diag(Sigma) @ VhE
    ax3.plot(SigmaVHS[0], SigmaVHS[1], 'b-')
    ax3.plot(SigmaVHE[0], SigmaVHE[1], 'g-')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    ax3.set_title("SigmaVHS")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis("equal")

    # Plot USigmaVHS
    ax4 = plt.subplot(224)
    USigmaVHS = U @ SigmaVHS
    USigmaVHE = U @ SigmaVHE
    ax4.plot(USigmaVHS[0], USigmaVHS[1], 'b-')
    ax4.plot(USigmaVHE[0], USigmaVHE[1], 'g-')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax4.set_title("USigmaVHS")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis("equal")

    # Show graph
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    # Raise ValueError if s > rank(A)
    if s > np.linalg.matrix_rank(A):
        raise ValueError ("s is greater than rank of A")

    # Compute the compact SVD
    U, Sigma, Vh = la.svd(A)

    # Store UTrunc, SigmaTrunc, VhTrunc
    UTrunc = U[:, :s]
    SigmaDiag = np.diag(Sigma)
    SigmaTrunc = SigmaDiag[:s, :s]
    VhTrunc = Vh[:s, :]

    # Find ATrunc
    ATrunc = UTrunc @ SigmaTrunc @ VhTrunc

    # Find Size
    Size = UTrunc.size + np.sqrt(SigmaTrunc.size) + VhTrunc.size

    # Return best rank s approximation of A, number of entries needed to store the truncated SVD
    return ATrunc, Size




# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Compute the compact SVD of A
    U, Sigma, Vh = la.svd(A)

    # Find Sigma Diagonal Matrix and Shape of Sigma Diagonal Matrix
    SigmaDiag = np.diag(Sigma)
    u, t = np.shape(SigmaDiag)

    # Raise Error if err is less than or equal to the singular value of A
    if (err <= Sigma[u-1]):
        raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank")

    # Initialize counter and Norm
    s = 0 
    Norm = Sigma[s]

    # Find lowest rank approximation of A with 2-norm error less than err
    while Norm > err and s < u - 1:
        s +=1
        Norm = Sigma[s]

    # Return lowest rank approximation and number of entries
    ATrunc, size = svd_approx(A, s)
    return ATrunc, size


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    
    # Load image
    image = imread(filename) / 255
    isGray = True
    
    # Check to see if the image is color or not
    if len(image.shape) == 3:
        isGray = False

    # If we have a gray image, do this:
    if isGray == True:

        # Find Initial Size; Find best rank-s approximation and Final Size
        InitialSize = np.size(image)
        ApproxImage, FinalSize = svd_approx(image,s)
        for a in ApproxImage:
            np.clip(a, 0,1)

        # Find difference in size
        Difference = InitialSize - FinalSize

        # Plot Original Image and Approximation
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(image, cmap = "gray")
        ax2.imshow(ApproxImage, cmap = "gray")
        ax1.set_title("Original")
        ax2.set_title("Approximation with Rank: " + str(s))
        plt.suptitle("Number of Entries Saved: " + str(Difference))
        plt.show()

    else:

        # Find Initial Size; Find  R, G, B Matrices
        InitialSize = np.size(image)
        R = image[:,:, 0]
        G = image[:,:, 1]
        B = image[:,:, 2]

        # Find R rank approximation
        ApproxRed, RedSize = svd_approx(R, s)
        for a in ApproxRed:
            np.clip(a, 0,1)

        # Find G rank approximation
        ApproxGreen, GreenSize = svd_approx(G, s)
        for a in ApproxGreen:
            np.clip(a, 0, 1)
        
        # Find B rank approximation
        ApproxBlue, BlueSize = svd_approx(B, s)
        for a in ApproxBlue:
            np.clip(a,0,1)

        # Generate Final Matrix; Find difference
        FinalMatrix = np.dstack((ApproxRed, ApproxGreen, ApproxBlue))
        Difference = InitialSize - (RedSize + GreenSize + BlueSize)

        # Plot original image and approximation
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(image)
        ax2.imshow(FinalMatrix)
        ax1.set_title("Original")
        ax2.set_title("Approximation with Rank: " + str(s))
        plt.suptitle("Number of Entries Saved: " + str(Difference))
        plt.show()
        








