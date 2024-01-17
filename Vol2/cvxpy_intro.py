# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name> David Camacho
<Class> ACME
<Date> 3/10/2023
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    # Create variable and objective function
    x = cp.Variable(3, nonneg = True)
    c = np.array([2,1,3])
    objective = cp.Minimize(c.T @ x)

    # Then, we'll write the constraints
    A = np.array([1,2,0])
    B = np.array([0,1,-4])
    C = np.array([2,10,3])
    P = np.eye(3)
    constraints = [A @ x <= 3, B @ x <= 1, C @ x >= 12, P @ x >= 0]

    # Create problem and solve it
    problem = cp.Problem(objective, constraints)
    value = problem.solve()
    return x.value, value


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """

    # Find length of matrix A
    n = len(A[0])

    # Initialize variable and objective 
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x,1))

    # Set out all the constraints
    constraints = [A@x == b]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    value = problem.solve()
    return x.value, value





    

def test2():

    A = np.array([[1,2,1,1], [0,3,-2,-1]])
    b = np.array([7,4])
    print(l1Min(A,b))


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    # Initialize the objective function
    x = cp.Variable(6, nonneg = True)
    c = np.array([4,7,6,8,8,9])
    objective = cp.Minimize(c.T @ x)

    # Write out the constraints
    const1 = [1,1,0,0,0,0]
    const2 = [0,0,1,1,0,0]
    const3 = [0,0,0,0,1,1]
    const4 = [1,0,1,0,1,0]
    const5 = [0,1,0,1,0,1]

    # Set out all the constraints
    P = np.eye(6)
    constraints = [const1 @ x == 7, const2 @ x == 2, const3 @ x == 4, const4 @ x == 5, const5 @ x == 8, P @ x >= 0]

    # Solve problem and return answer
    problem = cp.Problem(objective, constraints)
    value = problem.solve()
    return x.value, value


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Make Q, which is the quadratic term
    Q = np.array([[3,2,1], [2,4,2], [1,2,3]])

    # Make r, which is the linear term
    r = np.array([3,0,1])

    # Make objective function and constraints
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x,Q)+ r.T @ x))

    # Solve the problem and return answers
    value = prob.solve()
    return x.value, value

def examp2():

    Q = np.array([[4,2], [2,2]])
    r = np.array([1,-1])
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x,Q) + r.T @ x))
    #print(prob.solve())
    print(x.value)


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """

    # Find size of the matrix A
    row, col = np.shape(A)

    # Make objective function and constraints
    x = cp.Variable(col, nonneg = True)
    objective = cp.Minimize(cp.norm(A@x -b,2))

    # Make identity matrix
    P = np.eye(col)

    # Make constraints
    constraints = [sum(x) == 1, P @ x >= 0]

    # Solve the objctive function
    problem = cp.Problem(objective, constraints)
    value = problem.solve()

    # Return minimizer and minimal value
    return x.value, value

def examp5():
    A = np.array([[1,2,1,1], [0,3,-2,-1]])
    b = np.array([7,4])

    print(prob5(A,b))


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    

    # Read in the file food.npy
    file = np.load('food.npy', allow_pickle = True)

    # Multiply by the number of servings to get the nutrition value of the whole product
    for line in file:
        line[2:] = line[1]*line[2:]

    # Pull out different food items
    prices = file[:, 0]
    calories = file[:,2]
    fat = file[:,3]
    sugar = file[:,4]
    calcium = file[:,5]
    fiber = file[:,6]
    protein = file[:,7]

    # Initialize identity matrix
    P = np.eye(18)

    # Make objective function and constraints
    x = cp.Variable(18, nonneg = True)
    objective = cp.Minimize(prices.T @ x)
    constraints = [calories.T @ x <= 2000, fat.T @ x <= 65, sugar.T @ x <= 50, calcium.T @ x >= 1000, fiber.T @ x >= 25, protein.T @ x >= 46, P @ x >= 0]

    # Solve the objctive function
    problem = cp.Problem(objective, constraints)
    value = problem.solve()

    # Return minimizing vector and the total amount of money spent
    return x.value, value
