# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name> David Camacho
<Class> Vol 1
<Date> 10/17/22
"""

import numpy as np
from scipy import linalg as la
from imageio import imread
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy import sparse
import math


# Problem 1
def laplacian(A):

    # Sum rows
    diagonals = A.sum(axis = 1)
    
    # Create D
    D = np.diag(diagonals)

    # Return L
    return D - A


# Problem 2
def connectivity(A, tol=1e-8):

    # Find Laplacian
    L= laplacian(A)

    # Get real eigenvalues
    eigs = np.real(la.eigvals(L))
    for i in range(len(eigs)):
        if eigs[i] < tol:
            eigs[i] = 0

    # Find number of connected components; sort eigenvalues; Sort them
    connected_components = np.abs(np.count_nonzero(eigs) - len(eigs))
    eigs.sort()

    # Find Algebraic Connectivity
    Algebraic_connectivity = eigs[1]

    # Return desired values
    return connected_components, Algebraic_connectivity



# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):

        # Read image file
        image = imread(filename)

        # Scale Image
        self.scaled_image = image / 255

        # Find if the image is black and white or colored
        shape = image.shape
        if len(shape) == 3:
            self.black = False

            # Average the RGB values
            brightness = self.scaled_image.mean(axis = 2)
        else:

            self.black = True
            brightness = self.scaled_image

        # Flatten the matrix
        self.brightness = np.ravel(brightness)

        # Find dimensions
        self.m, self.n = self.scaled_image.shape[:2]


        
       

    # Problem 3
    def show_original(self):
        """ Display original image"""

        if self.black:
            plt.imshow(self.scaled_image, cmap = "gray")

        else:
            plt.imshow(self.scaled_image)

        plt.show()

        
    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        
        size = self.brightness.shape[0]

        # Initialize A
        A = sparse.lil_matrix((size, size))
        D = np.empty(size)
        shape = np.sqrt(self.scaled_image.size)

        # Compute weights
        for i in range(size):
            neighbors, distance = get_neighbors(i, r, self.m, self.n)
            B = np.abs(self.brightness[neighbors] - self.brightness[i])
            val = np.exp(-B/sigma_B2 - distance/sigma_X2)
            A[i, neighbors] = val
            D[i] = np.sum(val)

        # Convert A to csc_matrix
        A = A.tocsc()

        # Return desired values
        return A, D



    # Problem 5
    def cut(self, A, D):
        # Find Laplacian
        L = sparse.csgraph.laplacian(A)

        # Find D^(-1/2)
        D_half = sparse.diags(1/np.sqrt(D)).tocsc()

        # Find D^(-1/2)LD^(-1/2)
        answer = D_half @ L @ D_half
        
        # Get smallest eigenvalues and their eigenvectors
        eigs, vec = sparse.linalg.eigsh(answer, which = "SM", k = 2)

        # Reshape eigenvector
        vec = vec[:, 1]
        vec = vec.reshape((self.m, self.n))

        # Make mask
        mask = vec > 0
        return mask


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        # Get our A, D from adjacency
        A, D = self.adjacency()

        # Get our mask from cut()
        mask = self.cut(A,D)

        # If the image is black and white
        if self.black:

            # Plot the cut images
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(self.scaled_image*mask, cmap = "gray")
            ax2.imshow(self.scaled_image*~mask, cmap = "gray")
            ax3.imshow(self.scaled_image, cmap = "gray")
            plt.suptitle("Black and White Image")

        # If the image is colored
        else:

            # Plot the cut images
            update_mask = np.dstack((mask, mask, mask))
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(self.scaled_image*update_mask)
            ax2.imshow(self.scaled_image*~update_mask)
            ax3.imshow(self.scaled_image)
            plt.suptitle("Colored Image")

        # Show plot
        plt.show()


        
        

            



        

 
