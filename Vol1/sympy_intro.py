# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Name> David    
<Class> ACME
<Date> 1/17/22  
"""
import sympy as sy
import matplotlib.pyplot as plt
import numpy as np

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    # Make Symbols
    x = sy.symbols('x')
    y = sy.symbols('y')
    
    # Create and return expression
    Expression = sy.Rational(2,5)*(sy.exp(x**2 -y))*(sy.cosh(x+y)) + sy.Rational(3,7)*(sy.log(x*y +1))
    return Expression

# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """

    # Make Symbols
    x  = sy.symbols('x')
    i  = sy.symbols('i')
    j  = sy.symbols('j')
    
    # Create and return expression
    expr = sy.product(sy.summation(j* (sy.sin(x) + sy.cos(x)) ,(j, i, 5)), (i, 1, 5))
    return sy.simplify(expr)

# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-2,2]. Plot e^(-y^2) over the same domain for comparison.
    """

    # Make Symbols
    x, n = sy.symbols('x n')
    y = sy.symbols('y')

    # Create Maclaurin Series Expression with x = -y**2
    expr = sy.summation(x**n/(sy.factorial(n)), (n, 0, N))
    new_expr = expr.subs(x, -y**2)
    f = sy.lambdify(y, new_expr, "numpy")

    # Plot the truncated Maclaurin series and the actual function
    y = np.linspace(-2, 2, 100)
    plt.plot(y, f(y), 'b-', label = "Maclaurin Series")
    plt.plot(y, np.e**(-y**2), 'r-', label = "Actual Function")
    plt.title("Maclaurin Series vs. Actual Function")
    plt.xlabel("y")
    plt.ylabel("Function")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    
    # Make symbols
    x, y = sy.symbols('x y')
    r, theta = sy.symbols('r, theta')

    # Make expression and do subsitutions
    expr = 1 - (((x**2 + y**2)**sy.Rational(7,2) + 18*x**5*y - 60*x**3*y**3 + 18*x*y**5)/ (x**2 + y**2)**3)
    new_expr = expr.subs({x:r*sy.cos(theta), y:r*sy.sin(theta)})
    
    # Find Solutions, and lambdify the function 
    Solns = sy.solve(sy.simplify(new_expr), r)
    f = sy.lambdify(theta, Solns[0])

    # Plot Polar Coordinate Graph
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(f(t)*np.cos(t), f(t)*np.sin(t))
    plt.title("Polar Coordinate Graph")
    plt.xlabel("r(theta) * cos(theta)")
    plt.ylabel("r(theta) * sin(theta)")
    plt.show()



 



# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
   
    # Initialize symbols and dictionary
    x = sy.symbols('x')
    y = sy.symbols('y')
    lamb = sy.symbols('lamb')
    dict1 = dict()

    # Create Matrix and find Eigenvalues
    A = sy.Matrix([[x-y, x, 0], [x, x-y, x], [0, x, x-y]])
    LambdaEye = lamb*sy.eye(3)

    det = sy.det(A - LambdaEye)
    Eigs = sy.solve(det, lamb)

    # Compute eigenvectors
    for eig in Eigs:
        Null = (A - eig*sy.eye(3)).nullspace()
        dict1.update({eig: Null})

    # Return dictionary of eigenvalues and eigenvectors
    return dict1


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points over [-5,5]. Determine which
    points are maxima and which are minima. Plot the maxima in one color and the
    minima in another color. Return the minima and maxima (x values) as two
    separate sets.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """

    #Initialize set and symbols
    Minima = set()
    Maxima = set()
    x = sy.symbols('x')

    # Create polynomial and lambdify it
    Polynom = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x -100
    f = sy.lambdify(x, Polynom, "numpy")

    # Find Critical Points
    Deriv1 = sy.diff(Polynom)
    CriticalPoints = sy.solve(Deriv1, x)

    # Compute 2nd Derivative
    Deriv2 = sy.diff(Deriv1)

    # Find Local Minima and Maxima
    for cp in CriticalPoints:
        if Deriv2.evalf(subs = {x: cp}) > 0:
            Minima.add(cp)
        if Deriv2.evalf(subs = {x: cp}) < 0:
            Maxima.add(cp)

    # Plot p(x) with the minima and maxima marked
    x = np.linspace(-5,5,200)
    plt.plot(x, f(x), 'r-')
    for min in Minima:
        plt.plot(min, f(min), 'ko', label = "Minima")

    for max in Maxima:
        plt.plot(max, f(max), 'ro', label = "Maxima")

    plt.legend()
    plt.title("Polynomial")
    plt.show()

    # Return Minima and Maxima
    return Minima, Maxima



    


# Problem 7
def prob7():
    """Calculate the volume integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """

    # Initialize symbols and variables
    x,y,z = sy.symbols('x y z')
    r = sy.symbols('r')
    r1 = 2

    # Set Needed Variable Expressions
    Var1 = x*sy.sin(y)*sy.cos(z)
    Var2 = x*sy.sin(y)*sy.sin(z)
    Var3 = x*sy.cos(y)

    # Find Jacobian and Determinant of J
    f = sy.Matrix([Var1, Var2, Var3])
    J = f.jacobian([x,y,z])
    DetJ = sy.simplify(J.det())

    # Plot Volume vs. Radius
    s = np.linspace(0,3, 200)
    expr =  sy.integrate(DetJ* (Var1**2 + Var2**2 + Var3**2)**2, (x, 0, r), (z, 0, 2*sy.pi), (y, 0, sy.pi))
    f = sy.lambdify(r, expr)
    plt.plot(s, f(s), 'b-', label = "Volume of Sphere Function")
    plt.title("Volume Integral")
    plt.xlabel('Radius')
    plt.ylabel('Volume')
    plt.show()

    # Return answer when r = 2
    return(sy.integrate(DetJ* (Var1**2 + Var2**2 + Var3**2)**2, (x, 0, r1), (z, 0, 2*sy.pi), (y, 0, sy.pi))).evalf()

