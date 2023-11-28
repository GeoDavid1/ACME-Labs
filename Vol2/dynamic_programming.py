# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Name> David Camacho
<Class> ACME Lab
<Date> 4/6/23
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    
    # Initialize V dictionary
    V_dict = {}
    V_dict[N] = 0

    # Calcualte V(t) for all t
    for t in range(N-1, 0, -1):
        V_needed = V_dict[t+1]
        V_dict[t] = max(t/(t+1)*V_needed + 1/N, V_needed)

    # Find highest expected value and the optimal stopping point
    max_key = max(V_dict, key = V_dict.get)
    max_value = V_dict[max_key]

    # Return values
    return max_value, max_key


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    
    # Initialize lists
    num_list = []
    stopping_percentages = []
    max_probabilities = []


    # Find optinal stopping percentages and max probabilities
    for num in range(3, M+1):
        num_list.append(num)

        # Find optimal stopping percentages
        stop_percentage = calc_stopping(num)[1]/num
        stopping_percentages.append(stop_percentage)

        # Find max probabilities
        max_prob = calc_stopping(num)[0]
        max_probabilities.append(max_prob)


    # Plot data
    plt.plot(num_list, stopping_percentages, label = "Optimal Stopping Percentage")
    plt.plot(num_list, max_probabilities, label = "Maximum Probability")
    plt.title("Optimal Stopping Percentage and Max Probabilities")
    plt.xlabel("Number of Candidates")
    plt.ylabel("Percentage")
    plt.legend()
    plt.show()

    # Return the optimal stopping percentage for M
    return stopping_percentages[-1]



# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    
    # Initialize vector of divided cake
    w = []
    for i in range(0, N+1,1):
        w.append(i/N)

    
    # Initialize consumption matrix
    consump_matrix = np.zeros((N+1, N+1))

    # Create consumption matrix
    for row in range(N+1):
        for col in range(N+1):
            if row > col:
                consump_matrix[row, col] = u(w[row-col])

    
    # Return consumption matrix
    return consump_matrix



# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    
    # Initialize vector of divided cake
    col = []
    for i in range(0, N+1,1):
        col.append(u(i/int(N)))

    # Create matrix A
    A_temp = np.zeros((N+1, T))
    A = np.column_stack((A_temp, col))

    # Create matrix P
    P = np.zeros((N+1, T+1))


    # Initialize CV
    CV = np.zeros((N+1, T+1))

    # Starting from the next to last column, iterate backwards
    for t in range(T-1, -1, -1):

        # Initialize CV and update values
        temp_CV = np.zeros((N+1, T+1))
        for i in range(N+1):
            for j in range(T+1):
                temp_CV[i, j] = u(i/N - j/N) + B*A[j][t+1]

        
        
        # Make NaNs equal to 0
        temp_CV[np.isnan(temp_CV)] = 0

        # Find the largest value in each row of the current value matrix; fill in A
        for i in range(N+1):
            A[i][t] = np.max(temp_CV[i,:])

        for i in range(N+1):
            J = np.argmax(temp_CV[i,:])
            P[i][t] = i/N - J/N

        for i in range(N+1):
            P[i][T] = i/N

    return A, P



# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    
    # Get P and initialize the policy array
    P = eat_cake(T,N,B)[1]
    policy = []

    # Find optimal policy at each time
    row = N
    for i in range(0,T+1,1):
        val = P[int(row), i]
        policy.append(val)
        row = row - val * N

    # Return policy array
    return np.array(policy)

    




