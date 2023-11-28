# iPyParallel - Intro to Parallel Programming

from ipyparallel import Client
import numpy as np
import time 
import matplotlib.pyplot as plt

# Problem 1
def prob1():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    
    # Make client and dview
    client = Client()
    dview = client[:]
    
    # Execute import and close 
    dview.execute("import scipy.sparse as sparse")
    
    # Close client
    client.close()
    
    # Return the DirectView
    return dview


# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    
    # Make client and dview
    client = Client()
    dview = client[:]
    
    # Push the dictionary 
    dview.push(dx)
    
    # Remember to include blocking
    dview.block = True

    vars = dx.keys()
    
    for var in vars:
        print(dview[var])
        
        
    

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    

    # Define mean, min, max functions
    mean = lambda n: np.mean(np.random.standard_normal(n))
    min = lambda n: np.min(np.random.standard_normal(n))
    max = lambda n: np.max(np.random.standard_normal(n))
    
    # Set up client and dview
    client = Client()
    dview = client[:]
    
    # Get results for mean, min, and max functions
    mean_results = dview.apply_sync(mean, n)
    min_results = dview.apply_sync(min, n)
    max_results = dview.apply_sync(max, n)
    
    # Return lists of floats
    return mean_results, min_results, max_results
    
    
    
    

# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    
    # Initialize empty lists and N list
    parallel_times = []
    serial_times = []
    N = [1000000, 5000000, 10000000, 15000000]
    
    # Time the process parallely
    for n in N:
        start_time = time.time()
        prob3(n)
        end_time = time.time()
        time_elapsed = end_time - start_time
        parallel_times.append(time_elapsed)
    
    
    # Set up mean, min, and max functions
    mean = lambda n: np.mean(np.random.standard_normal(n))
    min = lambda n: np.min(np.random.standard_normal(n))
    max = lambda n: np.max(np.random.standard_normal(n))
    
    # Time the process serially
    for n in N:
        start_time = time.time()
        for engine in range(4):
            means, mins, maxs = mean(n), min(n), max(n)
        end_time = time.time()
        time_elapsed = end_time - start_time
        serial_times.append(time_elapsed)
        
    # Plot the data
    plt.plot(N, parallel_times, color = 'blue', label = 'Parallel Times')
    plt.plot(N, serial_times, color = 'red', label = 'Serial Times')
    plt.title('Problem 4: Serial and Parallel Execution Times')
    plt.xlabel('N')
    plt.ylabel('Time') 
    plt.legend()
    plt.show()   
            


# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    
    # Set up client and dview
    client = Client()
    dview = client[:]
    
    # Define h, X, and I
    h = (b - a) / (n-1)
    X = np.linspace(a, b, n).tolist()
    I = np.linspace(0,n-2,n-1).astype('int')
   
    # Define the sum function
    def sum1(i):
        return (h/2) * (f(X[i]) + f(X[i+1]))
    
    # Get the area for each part
    results = dview.map(sum1, I)
    
    # Get sum and return sum
    value = sum(results)
    return value
    

func = lambda x: 2
a = 1
b = 3

print(parallel_trapezoidal_rule(func, a,b))
    
    
    
    
    
    
    