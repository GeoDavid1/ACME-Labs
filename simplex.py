"""Volume 2: Simplex

<Name> David Camacho    
<Date> 3/1/23
<Class> ACME
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        
        # Check to see if the given system is infeasible at the origin
        if (0 < constraint for constraint in b):
            pass
        else:
            raise ValueError("System is infeasible at the origin.")
        
        # Set attributes
        self.dictionary = self._generatedictionary(c,A,b)
       



    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        #  Find length of A, length of c and make an identity matrix
        m = len(A)
        n = len(c)
        iden = np.identity(m)

        #  Make A_bar, zeros, and c_bar
        A_bar = np.column_stack((A, iden))
        zeros = np.zeros(m)
        c_bar = np.concatenate((c, zeros), axis = 0)
    
        # Create the dictionary
        first_part = np.concatenate((np.array([0]), b))
        second_part = np.row_stack((c_bar.T, -A_bar))
        D = np.column_stack((first_part, second_part))

        # Return the dictionary
        return D


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """

        # Get the top row, and matrix we care about
        

        m = self.dictionary.shape[1]
        for i in range(1,m):
            if self.dictionary[0,i]:
                return i


    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """

        
        ratios = []
        n = self.dictionary.shape[0]
        # use blands rule to calculate the ratios
        for i in range(1,n):
            if self.dictionary[i,index] < 0:
                ratios.append()



    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """

        # Find entry from previous functions and divide that row by -1* entry
        col = self._pivot_col()
        row = self._pivot_row(col)
        entry = self.dictionary[row, col]
        self.dictionary[row, :] /= -1* entry

        # Find number of columns
        n = len(self.dictionary)

        # Use row reducing to zero out the entries directly above and below the pivot point
        for i in range (0,n):
            if i != row:
                mult = self.dictionary[i, col]
                self.dictionary[i, :] +=  mult * self.dictionary[row, :]


    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """

        # Initialize dictionaries
        basic_dict = dict()
        nonbasic_dict = dict()

        # Pivot until top row is all positive
        while True in np.less(self.dictionary[0,1:],0):
            self.pivot()

        # Find minimal value and top row
        minimal_val = self.dictionary[0,0]
        top_row = self.dictionary[0, 1:]

        # Make two dictionaries
        for i in range(len(top_row)):

            # Make basic dictionaries
            if top_row[i] != 0:
                basic_dict[i] = 0
            
            # Make nonbasic dictionaries
            else:
                temp_col = self.dictionary[:, i+1]
                index = np.where(temp_col == -1)[0]
                val = float(self.dictionary[index, 0])
                nonbasic_dict[i] = val

        # Return values
        return (minimal_val, nonbasic_dict, basic_dict)





        
            






def test():

    c = np.array([-3,-2])
    A = np.array([[1,-1], [3,1], [4,3]])
    b = np.array([2,5,7])

    ss = SimplexSolver(c,A,b)
    ss._generatedictionary(c,A,b)
    index = ss._pivot_col()
    print(ss._pivot_row(index))
    #print(index)
    #print(ss._pivot_row(index))
    #print(ss.dictionary)
    #print(ss.pivot())
    print(ss.solve())


    #print(ss.solve())


import matplotlib.pyplot as plt
# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    
    # Load data
    data = np.load(filename)
    A = (data['A'])
    p = (data['p'])
    m = (data['m'])
    d = (data['d'])

    # Create a, b, c, matrices
    c = -p
    iden = np.eye(len(A[0]))
    a = np.vstack((A, iden))
    b = np.concatenate((m,d))

    # Solve system of equations
    Ss = SimplexSolver(c,a,b)
    dict =  Ss.solve()[1]

    # Return 1-D numpy array of number of units
    my_values = [dict[key] for key in dict.keys()]
    my_array = np.array(my_values)
    return my_array[0:4]

