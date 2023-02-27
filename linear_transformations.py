# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
import matplotlib.pyplot as plt
import time


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction."""

    # Multiply the matrix A by the stretching matrix

    Stretch_Matrix = np.array([[a, 0],[0, b]])

    Product = np.dot(Stretch_Matrix, A)

    # Return Product

    return Product

    
    raise NotImplementedError("Problem 1 Incomplete")

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction."""


    # Multiply the matrix A by the shearing matrix

    Shear_Matrix = np.array([[1, a], [b, 1]])

    Product = np.dot(Shear_Matrix, A)

    # Return Product

    return Product

    """Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix"""
    """
    raise NotImplementedError("Problem 1 Incomplete")"""

def reflect(A, a, b):
   
    # Create reflection matrix

   First_Matrix = np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])

   # Multiply the reflection matrix by the constant, the multiply the initial matrix by the reflection matrix

   Constant = 1/(a**2 + b**2)

   Reflect_Matrix = np.multiply(First_Matrix, Constant)

   Product = np.dot(Reflect_Matrix, A)

    # Return Product

   return Product


   

def rotate(A, theta):

    # Multiply the matrix A by the rotation matrrix

    Rotate_Matrix = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    Product = np.dot(Rotate_Matrix, A)

    # Return Product

    return Product




# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):

    # Initialize which values of T we are evaluating

    Time = np.linspace(0, T, 100)

    # Initialize arrays (1st array: x values, 2nd array: y values) of the position of the Earth and Moon

    EarthPositions= [[x_e], [0]]
    MoonPositions = [[x_m], [0]]

    # Run this loop for all time values in Time
    for time in Time[1:]:

        #Rotate the Earth by the angular rotation
        
        newEarthPosition = rotate([x_e, 0], time*omega_e)

        #Append x and y values of new Earth Position

        EarthPositions[0].append(newEarthPosition[0])
        EarthPositions[1].append(newEarthPosition[1])

        # Rotate the Moon by the angular rotation relative to the Earth

        newMoonPosition = rotate([x_m - x_e, 0], time*omega_m)

        # Append x and y values of the new Moon Position (not relative to Earth)

        MoonPositions[0].append(newMoonPosition[0] + newEarthPosition[0])
        MoonPositions[1].append(newMoonPosition[1] + newEarthPosition[1])


    #Plot the Data compiled by the arrays EarthPositions and MoonPositions to get desired graph
    
    plt.plot(EarthPositions[0], EarthPositions[1], label = "Earth")
    plt.plot(MoonPositions[0], MoonPositions[1], label = "Moon")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("Earth and Moon Positions vs. Time")
    plt.show()

    






   
    



def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():

    #Initialize Matrices

    TimesMatrixVector = []
    TimesMatrixMatrix = []

    # Initialize values that we are going to evaluate

    domain = 2**np.arange(1,9)

    # For all powers of 2, do this:

    for i in domain:

        # Create a randomVector (size i) and randomMatrix (size i x i)

        randomVector = random_vector(i)
        randomMatrix = random_matrix(i)

        # Find how much time it takes to multiply the vector and the matrix together
        start = time.time()
        Product = matrix_vector_product(randomMatrix, randomVector)
        end = time.time()
        timeTaken = end - start

        # Add time value to TimesMatrixVector

        TimesMatrixVector.append(timeTaken)

    # For all powers of 2, do this as well:

    for j in domain:

        # Create 2 randomMatric(es) (size i x i)

        randomMatrix1 = random_matrix(j)
        randomMatrix2 = random_matrix(j)

        # Find how much time it takes to multiply the matrices together

        start = time.time()
        Product = matrix_matrix_product(randomMatrix1, randomMatrix2)
        end = time.time()
        timeTaken = end - start

        # Add time value to TimesMatrixMatrix

        TimesMatrixMatrix.append(timeTaken)

    
    #Plot Time vs. Size of Vector and Matrix here (Subplot 1)
    
    ax1 = plt.subplot(121)
    ax1.plot(domain, TimesMatrixVector, 'b.-')
    ax1.set_xlabel("n")
    ax1.set_ylabel("seconds")
    plt.gca().set_xlim(left = 0)
    plt.gca().set_ylim(bottom = 0)
    ax1.set_title("Matrix-Vector Multiplication")


    #Plot Time vs. Size of Matrices here (Subplot 2)

    ax2 = plt.subplot(122)
    ax2.plot(domain, TimesMatrixMatrix, 'g.-')
    ax2.set_xlabel("n")
    ax2.set_ylabel("seconds")
    plt.gca().set_xlim(left = 0)
    plt.gca().set_ylim(bottom = 0)
    ax2.set_title("Matrix-Matrix Multiplication")


    # Show the two subplots

    plt.show()






# Problem 4
def prob4():

    #Initialize matrices

    TimesMatrixVector = []
    TimesMatrixMatrix = []
    DotMatrixVector = []
    DotMatrixMatrix = []

    # Initialize domain


    domain = 2**np.arange(1,9)

    for i in domain:

        # Create a randomVector (size i) and randomMatrix (size i x i)

        randomVector = random_vector(i)
        randomMatrix = random_matrix(i)

        # Find how much time it takes to multiply the vector and the matrix together (not Numpy)

        start = time.time()
        Product = matrix_vector_product(randomMatrix, randomVector)
        end = time.time()
        timeTaken = end - start

        # Add time value to TimesMatrixVector

        TimesMatrixVector.append(timeTaken)

    for j in domain:

        # Create 2 randomMatric(es) (size i x i)

        randomMatrix1 = random_matrix(j)
        randomMatrix2 = random_matrix(j)

        # Find how much time it takes to multiply the matrices together (not Numpy)

        start = time.time()
        Product = matrix_matrix_product(randomMatrix1, randomMatrix2)
        end = time.time()
        timeTaken = end - start

        # Add time value to TimesMatrixMatrix

        TimesMatrixMatrix.append(timeTaken)

    for k in domain:

        # Create a randomVector (size i) and randomMatrix (size i x i)

        randomVector = random_vector(k)
        randomMatrix = random_matrix(k)

        # Find how much time it takes to multiply the vector and the matrix together (Numpy)

        start = time.time()
        Product = np.dot(randomMatrix, randomVector)
        end = time.time()
        timeTaken = end - start

        # Add time value to DotMatrixVector

        DotMatrixVector.append(timeTaken)

    for m in domain:

        # Create 2 randomMatric(es) (size i x i)

        randomVector = random_vector(m)
        randomMatrix = random_matrix(m)

        # Find how much time it takes to multiply the matrices together (Numpy)

        start = time.time()
        Product = np.dot(randomMatrix, randomVector)
        end = time.time()
        timeTaken = end - start

        # Add time value to DotMatrixMatrix

        DotMatrixMatrix.append(timeTaken)



    #Plot for linear graphs

    ax1 = plt.subplot(121)

    # Plot four different desired graphs

    ax1.plot(domain, TimesMatrixVector, 'b.-', label= 'Matrix-Vector')
    ax1.plot(domain, TimesMatrixMatrix, 'g.-', label = 'Matrix-Matrix')
    ax1.plot(domain, DotMatrixVector, 'm.-', label = 'Dot Matrix-Vector')
    ax1.plot(domain, DotMatrixMatrix, 'k.-', label = 'Dot Matrix-Matrix')

    # Set parameters for subplot

    ax1.set_xlabel("n")
    ax1.set_ylabel("seconds")
    plt.gca().set_xlim(left = 0)
    plt.gca().set_ylim(bottom = 0)
    ax1.set_title("Linear Plot")
    ax1.legend()

    # Plot for logarithmic graphs

    ax2 = plt.subplot(122)

    # Plot four different desired graphs

    ax2.loglog(domain, TimesMatrixVector, 'b.-', base=2, lw = 2, label = 'Matrix-Vector')
    ax2.loglog(domain, TimesMatrixMatrix, 'g.-', base=2, lw = 2, label = 'Matrix-Matrix')
    ax2.loglog(domain, DotMatrixVector, 'm.-', base=2, lw = 2, label = 'Dot Matrix-Vector')
    ax2.loglog(domain, DotMatrixMatrix, 'k.-', base=2, lw = 2, label = 'Dot Matrix-Matrix')

    # Set parameters for subplot

    ax2.set_xlabel("n")
    ax2.set_ylabel("seconds")
    ax2.set_title("Logarithmic Plot")
    ax2.legend()


    # Show subplots

    plt.show()




    


