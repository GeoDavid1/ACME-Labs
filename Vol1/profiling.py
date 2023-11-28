# profiling.py
"""Python Essentials: Profiling.
<Name> David Camacho
<Class> ACME VOL 1
<Date> 3/1/23
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from numba import jit
import time
from matplotlib import pyplot as plt



# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    
    # Open file
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    
    # Iterate from bottom to top adding the max from the bottom
    for n in range(-2, -1*len(data) - 1, -1):
        add_to_line = data[n]
        adding_line = data[n+1]
        for i in range(0,len(add_to_line),1):
            add_to_line[i] = max(add_to_line[i] + adding_line[i], add_to_line[i] + adding_line[i+1])
        data[n] = add_to_line
    
    # Return the max sum
    return(data[0][0])


# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list


def primes_fast(N):
    """Compute the first N primes."""
    
    # Start primes list and start going up
    primes_list = [2]
    candidate = 3 
    integer = 0

    # Go through all the prime divisors
    while(len(primes_list) < N):
        is_prime = True

        # For each prime divisor
        for divisor in primes_list[1:integer]:
            # Break if candidate is not prime
            if candidate % divisor == 0:
                is_prime = False
                break
        # If candidate is prime add to list
        if is_prime:
            primes_list.append(candidate)
        candidate += 2

        # Check if divisor is less than sqaure root of candidate
        if primes_list[integer]**2 <= candidate:
                integer += 1

    # Return list of primes
    return primes_list



# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    # Compute closest array using array broadcasting
    return np.argmin(np.linalg.norm(A - x.reshape((A.shape[0],1)), axis = 0))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    
    # Make a dictionary of the alphabet
    alphabet_dict = {chr(i): i - 64 for i in range(65,91)}
    
    # Open file and sort names
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))

    # Initialize variables
    total = 0
    k = 0

    # Go through each name
    for item in names:
        name_value = 0
        # Go through each letter in each name and sum up score for name
        for letter in item:
            letter_value = alphabet_dict[letter]
            name_value += letter_value
        # Find total score
        total += (k+1)* name_value
        k += 1
    # Return total score
    return total





# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    
    # Initialize the terms of Fibonacci sequence
    x1 = 1
    x2 = 1
    yield 1
    yield 1
    
    # Run indefinitely
    while True:

        # Calculate new Fibonacci term
        temp = x2
        x2 = x2 + x1
        x1 = temp
        
        # Yield new Fibonacci term
        yield x2

def test5():

    for i in fibonacci():
        print(i)
        if i>2:
            break

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    # Go through the Fibonacci sequence generator until the term has length N
    for i, x in enumerate(fibonacci()):
        if len(str(x)) == N:
            # Return the index
            return i  + 1
            break


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
   
    # Initialize primes list
    list = np.arange(2, N+1)

    # Clean out the composite numbers
    while len(list) > 0:
        yield list[0] 
        list = list[list % list[0] != 0]


def test6():

    for i in prime_sieve(100):
        print(i)
        

# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    
    # Make a copy and temporary array; find shape
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    
    # Run this code to find the nth power
    for power in range(1,n):
        # Perform matrix multiplication
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            # Save new matrix
            product[i] = temporary_array
    # Return matrix power
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    # Initialize time lists
    normal_times = []
    numba_times = []
    linalg_times = []

    # Create sizes
    sizes = np.array([2**pow for pow in range(2,8,1)])

    # Run matrix_power_numba() once
    B = np.random.random((2,2))
    rand_compile = matrix_power_numba(B,3)

    # Time for all sizes
    for size in sizes:
        
        # Make a random A of size size
        A = np.random.random((size, size))

        # Find execution times for normal function
        normal_start = time.time()
        test1 = matrix_power(A,n)
        normal_end = time.time()
        normal_time = normal_end - normal_start
        normal_times.append(normal_time)

        # Find execution times for numba function
        numba_start = time.time()
        test2 = matrix_power_numba(A,n)
        numba_end = time.time()
        numba_time = numba_end - numba_start
        numba_times.append(numba_time)

        # Find execution times for np.linalg function
        linalg_start = time.time()
        test3 = np.linalg.matrix_power(A,n)
        linalg_end = time.time()
        linalg_time = linalg_end - linalg_start
        linalg_times.append(linalg_time)


    # Plot results
    plt.loglog(sizes, normal_times, label = "Normal Times")
    plt.loglog(sizes, numba_times, label = "Numba Times")
    plt.loglog(sizes, linalg_times, label = "NP.Linalg Times")
    plt.xlabel("Log Size")
    plt.ylabel("Log Time")
    plt.title("Log Times vs. Log Sizes")
    plt.legend()
    plt.show()







@jit
def row_sum_numba(A):
    """Sum the rows of A by iterating through rows and columns, optimized by Numba"""
    
    m,n = A.shape
    row_totals = np.empty(m)
    for i in range(m):
        total = 0
        for j in range(n):
            total += A[i,j]
        row_totals[i] = total
    return row_totals

from numba import int64, double

@jit(nopython = True, locals = dict(A = double[:,:], m = int64, n = int64, row_totals = double[:], total = double))

def row_sum_numba(A):
    m,n = A.shape
    row_totals = np.empty(m)
    for i in range(m):
        total = 0
        for j in range(n):
            total += A[i,j]
        row_totals[i] = total
    return row_totals
