# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name> David Camacho
<Class>
<Date> 9/15/22
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):

    Mean_Array = []                                           
    Gaussian_Matrix = np.random.normal(size=(n,n))             # Create a random matrix of size n x n
    Means = np.mean(Gaussian_Matrix, axis =1)                  
    Variance_Means = np.var(Means)                             # Calculate Variance of Means
    

    return Variance_Means


def prob1():

    Variance_Values = []

    # Append to array of Variances, the variance of means of matrix size i x i

    for i in range(100, 1001, 100):                           
        print(i)
        Variance_Values.append(var_of_means(i))

    # Set x-axis
    
    x = np.linspace(100, 1000, 10) 

    # Plot/show data and make labels                           

    plt.plot(x, Variance_Values)                              
    plt.title("Problem 1: Variance of Means")       
    plt.xlabel("n")
    plt.ylabel("Variance of The Means")
    plt.show()

# Problem 2
def prob2():

    #Initialize x-axis and functions sin(x), cos(x), arctan(x)

    x = np.linspace(-1*2*(np.pi), 2*(np.pi), 100)     
    a = np.sin(x)
    b = np.cos(x)
    c = np.arctan(x)

    #Plot functions and title graph and axes

    plt.plot(x, a)
    plt.plot(x, b)
    plt.plot(x, c)

    # Label, title, and show plot
    
    plt.title("Problem 2: sin(x) vs. cos(x) vs. arctan(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Problem 3
def prob3():

    # Set 2 domains for f(x)

    x1 = np.linspace(-2, 0.9999, 100)
    x2 = np.linspace(1.00001, 6, 100)[1:]

    # Plot the function f(x)

    plt.plot(x1, 1/(x1-1), 'm--', linewidth=4)
    plt.plot(x2, 1/(x2-1), 'm--', linewidth =4)

    # Set range of x-axis and y-axis

    plt.xlim([-2, 6])
    plt.ylim([-6,6])

    # Label title and axes

    plt.title("Problem 3: f(x) = 1/(x-1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Problem 4
def prob4():

    # Set limits of each setplot and initialize x-axis
    plt.axis([0, 2*(np.pi), -2, 2])
    x1 = np.linspace(0, 2*(np.pi), 100)

    # Plot, title, and label sin(x)

    ax1 = plt.subplot(221)
    ax1.plot(x1, np.sin(x1), 'g-')
    plt.xlim(0, 2*np.pi)
    plt.ylim(-2, 2)
    ax1.set_title("sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot, title and label sin(2x)

    
    ax2 = plt.subplot(222)
    ax2.plot(x1, np.sin(2*x1), 'r--')
    plt.xlim(0, 2*np.pi)
    plt.ylim(-2, 2)
    ax2.set_title("sin(2x)")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot, title, and label 2sin(x)

    
    ax3 = plt.subplot(223)
    ax3.plot(x1, 2*np.sin(x1), 'b--')
    plt.xlim(0, 2*np.pi)
    plt.ylim(-2, 2)
    ax3.set_title("2sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot, title, and label 2sin(2x)

    
    ax4 = plt.subplot(224)
    ax4.plot(x1, 2*(np.sin(2*x1)), 'm:')
    plt.xlim(0, 2*np.pi)
    plt.ylim(-2,2)
    ax4.set_title("2sin(2x)")
    plt.xlabel("x")
    plt.ylabel("y")

    # Title entire graph and show plot

    plt.suptitle("Problem 4: Plots of functions related to sin(x)")
    plt.tight_layout()
    plt.show()


# Problem 5
def prob5():


    # Load data from FARS

    FARS= np.load("FARS.npy")

    # Scale axes and extract needed data from FARS

    plt.axis("equal")
    Military_Time_Array = FARS[:,0]
    Longitudes_Array = FARS[:,1]
    Latitudes_Array = FARS[:,2]

    # Plot, title, and label scatterplot of longitudes against latitudes

    plt.subplot(1,2,1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.plot(Longitudes_Array, Latitudes_Array, 'ok', markersize = 5, alpha = 0.5)
    plt.title("Longitude and Latitude of Car Crashes in FARS")

    # Plot, title, and label histogram of time of crashes

    plt.subplot(1,2,2)
    plt.xlabel("Time (Military Time)")
    plt.ylabel("Number of Crashes")
    plt.title("Times of Crashes in FARS")
    plt.hist(Military_Time_Array, bins = np.arange(0, 25,1))


    # Title entire plot and show plot

    plt.suptitle("Problem 5: FARS Data")
    plt.show()



# Problem 6
def prob6():

    # Initialize initial values and make MeshGrid

    x = np.linspace(-2 *(np.pi), 2* (np.pi), 100)
    y = x.copy()
    X, Y = np.meshgrid(x,y)
    Z = (np.sin(X) * np.sin(Y))/(X*Y)

    # Title, and label heat map of g(x,y)

    plt.subplot(1,2,1)
    plt.title("Heat Map of g(x,y) = sin(x)sin(y)/xy")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot heat map of g(x,y) with color bar

    plt.pcolormesh(X, Y, Z, cmap= "viridis", shading = "auto")
    cbar = plt.colorbar()
    cbar.set_label("z")

    # Set limits of subplot

    plt.xlim(-1*2* (np.pi), 2* (np.pi))
    plt.ylim(-1*2* (np.pi), 2* (np.pi))

    # Title, and label contour map of g(x,y)

    plt.subplot(1,2,2)
    plt.title("Contour Map of g(x,y) = sin(x)sin(y)/xy")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot contour map of g(x,y) with color bar

    dbar = plt.colorbar()
    dbar.set_label("z")
    plt.contour(X,Y,Z, 30, cmap = "magma")

    # Set limits of subplot

    plt.xlim(-1*2* (np.pi), 2* (np.pi))
    plt.ylim(-1*2* (np.pi), 2* (np.pi))

    # Title entire plot and show plot

    plt.suptitle("Problem 6: Heat Map and Contour Map of g(x,y) = sin(x)sin(y)/xy")
    plt.show()



