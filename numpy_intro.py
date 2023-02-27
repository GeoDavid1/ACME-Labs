# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Name>
<Class>
<Date>
"""
import numpy as np

def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    
    A = np.array([[3, -1, 4], [1,5,-9]])  #Define array A

    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])  #Define 

    new_matrix = np.zeros((2,4))

    


    Num_Rows_A = len(A)   #Find # of rows of A
    Num_Columns_B = len(B[0]) # Find # of columns of B

 
    i = 0    #Initialize counters
    j = 0

    while i < Num_Rows_A:         #Iterate through all the rows of A
        j = 0                     #Reset the column counter to go through all the columns again
        while j < Num_Columns_B:  #Iterate throrugh all the columns of B
            new_matrix[i, j] = np.dot(A[i], B[:,j])   #Find the dot product of every combination of row in A and column in B
            j += 1              # Go up one column in B

        i +=1                   # Go up one row in A

    return new_matrix           # Return desired value
    
    #raise NotImplementedError("Problem 1 Incomplete")

#print(prob1())



def prob2():

    A = np.array([[3, 1, 4], [1,5,9], [-5,3,1]])   #Initialize matrix A

    new_A_squared_matrix = np.zeros((3,3))   #Initialize a matrix full of zeros for A squared
    new_A_cubed_matrix = np.zeros((3,3))     #Initialize a matrix full of zeros for A cubed

    Num_Rows_A = len(A)  #Find Number of Rows of A
    Num_Columns_A = len(A[0])  #Find Number of Columns of A

    i = 0   #Initialize counters
    j = 0

    while i < Num_Rows_A:  #Iterate through all the rows of A
        j = 0              #Reset the column counter to go through all the columns again
        while j < Num_Columns_A:        #Iterate through all the columns of A
            new_A_squared_matrix[i,j] = np.dot(A[i], A[:, j])    #Find the dot product of every combination of row in A and column in B
            j += 1   #Go up one column in A

        i +=1  #Go up one row in A

    k = 0     #Initialize counters
    l = 0


    while k < Num_Rows_A:   #Iterate through all the rows of A
        l = 0   #Reset the column counter to go through all the columns again
        while l < Num_Columns_A:   #Iterate through all the columns of new_A_squared_matrix
            new_A_cubed_matrix[k,l] = np.dot(A[k], new_A_squared_matrix[:, l])    #Find the dot product of every combination of row in A and column in new_A_squared_matrix
            l += 1    #Go up one column in new_A_squared_matrix

        k += 1   #Go up one row in A

    return(-1*(new_A_cubed_matrix) + 9*(new_A_squared_matrix) -15*(A))  #Return the desired result
    
    



    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    #raise NotImplementedError("Problem 2 Incomplete")

#print(prob2())
def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """

    One_A = np.ones((7,7))             #Initialize a matrix full of 1's
    Initial_B = np.full((7,7), -6)     # Initialize a matrix full of -6's

    Final_A = np.triu(One_A)                        # Make the Matrix A
    Lower_Triangular_B = np.tril(Initial_B)         # Make the lower_triangular matrix of B

    Final_B = Lower_Triangular_B + 5                # Make the Matrix B

    A_B = np.dot(Final_A, Final_B)                  # Calculate A*B

    A_B_A = np.dot(A_B, Final_A)                    # Calculate A*B*A


    #print(type(A_B_A))
    return (A_B_A.astype(np.int64))                                    # Return A*B*A
    
    #raise NotImplementedError("Problem 3 Incomplete")

#print(prob3())

def prob4(A):

    new_Array = np.array(A.copy()) # Copy the given array and make sure it is an array


    mask = new_Array < 0  # Set mask to make all values less than 0 have a bool value of True
    new_Array[mask] = 0  #Set all values with a bool values of True (less than 0) to 0

    return new_Array    # Return desired result


    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    #raise NotImplementedError("Problem 4 Incomplete")


def prob5():

    A = np.array([[0,2,4], [1,3,5]])                   #Initialize arrays
    B = np.array([[3,0,0], [3,3,0], [3,3,3]])
    C = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])

    A_Transpose = np.copy(A)                            # Copy Matrix A                    
    A_Transpose = A.T                                   # Make the transpose of A

    Identity_Matrix = np.eye(3)                         #Create identity matrix

    Three_By_Three_Zero = np.zeros((3,3))              #Initialize zero arrays
    Two_By_Two_Zero = np.zeros((2,2))
    Three_By_Two_Zero = np.zeros((3,2))
    Two_By_Three_Zero = np.zeros((2,3))

    First_Stack = np.vstack((Three_By_Three_Zero, A, B))                               #Create stacks
    Second_Stack = np.vstack((A_Transpose, Two_By_Two_Zero, Three_By_Two_Zero))
    Third_Stack = np.vstack((Identity_Matrix, Two_By_Three_Zero, C))

    Final_Stack = np.hstack((First_Stack, Second_Stack, Third_Stack))                  #Make final stack

    #print(First_Stack)
    #print(Second_Stack)
    #print(Third_Stack)

    return Final_Stack                          #Return desired product

    #print(A_Transpose)
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    #raise NotImplementedError("Problem 5 Incomplete")

#print(prob5())

def prob6(A):
    sum_array = A.sum(axis =1)    #Find sums of each row

    return(((A.T)/(sum_array)).T)           # Return the row-stochastic matrix


    

    

    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.


    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    #raise NotImplementedError("Problem 6 Incomplete")

#A = np.array([[4,5,6], [1,2,4]])
#print(prob6(A))

def prob7():

    Largest_Product = 0

    grid = np.load("grid.npy")     #Load the grid

    Horizontal_Max = np.max(grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:])  # Find the maximum Horizontal sum

    Vertical_Max = np.max(grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :])    # Find the maximum Vertical sum

    Diagonal_Max_1 = np.max(grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:]) # Find the maximum Diagonal sum

    Diagonal_Max_2 = np.max(grid[:-3, 3: ] * grid[1:-2, 2:-1] * grid[2:-1, 1:-2] * grid[3:, :-3]) # Find the maximum Diagonal sum (other direction)


    

    return(max(Horizontal_Max, Vertical_Max, Diagonal_Max_1, Diagonal_Max_2))   #Return max of all the max's

#print(prob7())





    
    #raise NotImplementedError("Problem 7 Incomplete")
