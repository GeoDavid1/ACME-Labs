# differentiation.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""
import sympy as sy
import numpy as np
from matplotlib import pyplot as plt
from jax import numpy as jnp
from jax import grad
import random
import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    
    # Create expression
    x = sy.symbols('x')
    Expr = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))

    # Take symbolic derivative and lambdify the resulting function
    diff = sy.diff(Expr, x)
    f = sy.lambdify(x, diff, "numpy")
    return f

# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""

    # Calculate the first order forward difference quotient
    deriv = (f(x + h) - f(x))/(h)
    return deriv
    
def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""

    # Calculate the second order forward difference quotient
    deriv = (-3*f(x) + 4*f(x+h) - f(x + 2*h))/(2*h)
    return deriv

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""

    # Calculate the first order backward difference quotient
    deriv = (f(x) - f(x-h))/(h)
    return deriv
    

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""

    # Calculate the second order backward difference quotient
    deriv = (3*f(x) - 4*f(x-h) + f(x - 2*h))/(2*h)
    return deriv
    
def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""

    # Calculate the second order centered difference quotient
    deriv = (f(x + h) - f(x - h))/(2*h)
    return deriv
    
def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""

    # Calculate the fourth order centered difference quotient
    deriv = (f(x - 2*h) - 8 * f(x- h) + 8*f(x+h) - f(x + 2*h))/(12*h)
    return deriv
    

# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """

    # Lambdify f
    f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
    
    # Find exact value of derivative
    exact = prob1()(x0)
    
    # Initialze h values and Error Arrays
    hArray = np.logspace(-8, 0, 9)
    O1FError = []
    O2FError = []
    O1BError = []
    O2BError = []
    O2CError = []
    O4CError = []

    # Calculate absolute values for all functions in Problem 2
    for H in hArray:

        O1FError.append(np.abs(exact - fdq1(f, x0, H)))
        O2FError.append(np.abs(exact - fdq2(f, x0, H)))
        O1BError.append(np.abs(exact - bdq1(f, x0, H)))
        O2BError.append(np.abs(exact - bdq2(f, x0, H)))
        O2CError.append(np.abs(exact - cdq2(f, x0, H)))
        O4CError.append(np.abs(exact - cdq4(f, x0, H)))
        
    # Plot results
    plt.loglog(hArray, O1FError, 'b.-', label = "Order 1 Forward")
    plt.loglog(hArray, O2FError, 'g.-', label = "Order 2 Forward")
    plt.loglog(hArray, O1BError, 'r.-', label = "Order 1 Backward")
    plt.loglog(hArray, O2BError, 'c.-', label = "Order 2 Backward")
    plt.loglog(hArray, O2CError, 'k.-', label = "Order 2 Centered")
    plt.loglog(hArray, O4CError, 'y.-', label = "Order 4 Centered")
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs. h")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """

    # Load data; initialze arrays
    plane_data = np.load("plane.npy")
    x_coord = []
    y_coord = []
    x_prime = []
    y_prime = []

    # Convert alpha and beta to radians
    for time in plane_data:
        time[1] = np.deg2rad(time[1])
        time[2] = np.deg2rad(time[2])

    # Calculate x and y values at different times
    for time in plane_data:
        alpha = time[1]
        beta = time[2]
        x = (500) * (np.tan(beta)/(np.tan(beta) - np.tan(alpha)))
        y = (500) * (np.tan(beta) * np.tan(alpha))/(np.tan(beta) - np.tan(alpha))
        x_coord.append(x)
        y_coord.append(y)

    # Create a list of lists of time for convinence
    times = []
    for i in range(0,8):
        times.append(plane_data[i])

    # Extract data 
    t7, t8, t9, t10, t11, t12, t13, t14 = times
    x7, x8, x9, x10, x11, x12, x13, x14 = x_coord
    y7, y8, y9, y10, y11, y12, y13, y14 = y_coord

    # Calculate x prime at t = 7
    x_prime7 = x8 - x7
    y_prime7 = y8 - y7

    # Calculate x prime at t = 14
    x_prime14 = x14 - x13
    y_prime14 = y14 - y13

    # Calculate x prime at t = 8
    x_prime8 = (x9 - x7)/2
    y_prime8 = (y9 - y7)/2

    # Calculate x prime at t = 9
    x_prime9 = (x10 - x8)/2
    y_prime9 = (y10 - y8)/2

    # Calculate x prime at t = 10
    x_prime10 = (x11 - x9)/2
    y_prime10= (y11 - y9)/2

    # Calculate x prime at t = 11
    x_prime11 = (x12 - x10)/2
    y_prime11 = (y12 - y10)/2

    # Calculate x prime at t = 12
    x_prime12 = (x13 - x11)/2
    y_prime12 = (y13 - y11)/2

    # Calculate x prime at t = 13
    x_prime13 = (x14 - x12)/2
    y_prime13 = (y14 - y12)/2

    # Combine into arrays
    x_prime.extend((x_prime7, x_prime8, x_prime9, x_prime10, x_prime11, x_prime12, x_prime13, x_prime14))
    y_prime.extend((y_prime7, y_prime8, y_prime9, y_prime10, y_prime11, y_prime12, y_prime13, y_prime14))

    # Convert into np arrays
    x_array = np.array(x_prime)
    y_array = np.array(y_prime)

    # Create and return speed array
    speed = np.sqrt((x_array)**2 + (y_array)**2)
    return speed


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """

    # Initialize Jacobian matrix
    n = np.size(x)
    m = np.size(f(x))
    J = np.zeros((m,n))

    # Initialize identity matrix
    iden = np.identity(n)

    # Find all the partial derivatives
    for i in range(0,m):
        for j in range(0,n):
            deriv = (f(x + h*iden[:, j])[i] - f(x - h*iden[:, j])[i])/(2*h)
            J[i][j] = deriv

    # Return Jacobian matrix
    return J

    



# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    
    # 0th Chebyshev Polynomial
    if n == 0:
        return jnp.ones_like(x)
    
    # 1st Chebyshev Polynomial
    elif n == 1:
        return x

    # Recursively compute T_n(x)
    else:
        return 2*x*cheb_poly(x,n-1) - cheb_poly(x, n-2)



    

def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    
    # Compute T_prime_0
    funct0 = lambda x: cheb_poly(x,0)
    d_funct0 = jnp.vectorize(grad(funct0))

    # Compute T_prime_1
    funct1 = lambda x: cheb_poly(x,1)
    d_funct1 = jnp.vectorize(grad(funct1))

    # Compute T_prime_2
    funct2 = lambda x: cheb_poly(x,2)
    d_funct2 = jnp.vectorize(grad(funct2))

    # Compute T_prime_3
    funct3 = lambda x: cheb_poly(x,3)
    d_funct3 = jnp.vectorize(grad(funct3))

    # Compute T_prime_4
    funct4 = lambda x: cheb_poly(x,4)
    d_funct4 = jnp.vectorize(grad(funct4))

    # Plot Derivatives of Chebyshev polynomials
    domain = np.linspace(-1,1,100)
    plt.plot(domain, d_funct0(domain), 'b-', label = "0th Chebyshev")
    plt.plot(domain, d_funct1(domain), 'g-', label = "1st Chebyshev")
    plt.plot(domain, d_funct2(domain), 'r-', label = "2nd Chebyshev")
    plt.plot(domain, d_funct3(domain), 'c-', label = "3rd Chebyshev")
    plt.plot(domain, d_funct4(domain), 'k-', label = "4th Chebyshev")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Derivatives of Chebyshev Polynomials")
    plt.legend()
    plt.show()


# Problem 7
def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """

    # Lambdify f in JAX
    f = lambda x: (jnp.sin(x) + 1)**(jnp.sin(jnp.cos(x)))

    # Initialize arrays
    sympy_times = []
    quotient_times = []
    quotient_errors = []
    jax_times = []
    jax_errors = []

    # Perform the experiment N times
    for i in range(0,N):
        
        # Choose a random value x
        x = random.uniform(0, 2*np.pi)

        # Time how long it takes to calculate the "exact" value
        start1_time = time.time()
        exact = prob1()(x)
        end1_time = time.time()
        time1 = end1_time - start1_time
        sympy_times.append(time1)

        # Time how long it takes to calculate the fourth-order centered difference quotient
        start2_time = time.time()
        fourth_order = cdq4(f,x)
        end2_time = time.time()
        time2 = end2_time - start2_time
        quotient_times.append(time2)

        # Record absolute errors for fourth-order centered difference quotient
        err = np.abs(fourth_order - exact)
        quotient_errors.append(err)

        # Time how long it takes to calculate the approximation using JAX
        start3_time = time.time()
        jax_approx = grad(f)(x)
        end3_time = time.time()
        time3 = end3_time - start3_time
        jax_times.append(time3)

        # Record absolute errors for approximation using JAX
        err = np.abs(jax_approx - exact)
        jax_errors.append(time3)

    # Make array of Sympy errors
    sympy_errors = []
    for j in range(0,N):
        sympy_errors.append(1 * 10**-18)

    # Plot results
    plt.loglog(sympy_times, sympy_errors ,'bo', base = 10, label = "SymPy" )
    plt.loglog(quotient_times, quotient_errors, 'yo', base = 10, label = "Difference Quotients")
    plt.loglog(jax_times, jax_errors, 'ro', base = 10, label = "JAX")
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Errors vs. Computation Time")
    plt.legend()
    plt.show()









    
