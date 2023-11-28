# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name> David Camacho
<Class> ACME
<Date> 4/13/23
"""

import scipy.linalg as la
import numpy as np

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    
    # Initialize V arrays and counter
    V_old = np.zeros(nS)
    V_new = np.zeros(nS)
    k = 0

    # Loop while k < maxiter
    while k < maxiter:
        # Do it for all states
        for s in range(nS):
            sa_vector = np.zeros(nA)
            # Loop through all actions
            for a in range(nA):
                for tuple_info in P[s][a]:
                    # Get tuple
                    p, s_, u, _ = tuple_info

                    # Sums up the possible end states and rewards with given action
                    sa_vector[a] += (p * (u + beta * V_old[s_]))

            # Add the max value to the value function
            V_new[s] = np.max(sa_vector)

        # Break if V converges
        if la.norm(V_new - V_old) < tol:
            break

        # Make V_old V_new, increment counter
        V_old = V_new.copy()
        k += 1

    # Return final vector representing V and the number of iterations
    return V_old, k + 1

def test1():
    print(value_iteration(P, 4,4))



# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """

    # Initialize policy vector
    V_new = np.zeros(nS)
   
    # Loop through all states
    for s in range(nS):
        sa_vector = np.zeros(nA)
        # Loop through all actions
        for a in range(nA):
            for tuple_info in P[s][a]:
                # Get tuple
                p, s_, u, _ = tuple_info

                # Sums up the possible end states and rewards with given action
                sa_vector[a] += (p * (u + beta * v[s_]))

        # Find the argmax of the state
        V_new[s] = np.argmax(sa_vector)


    # Reset v
    v = V_new.copy()


    # Return the policy vector corresponding to V
    return V_new

def test2():
    print(extract_policy(P, 4,4,np.array([1,1,1,0])))

   


# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """

    # Initialize V old
    V_old = np.zeros(nS)

    # Loop until we reach a tolerance of the function
    while True:
        V_new = np.zeros(nS)

        # Loop through each state
        for s in range(nS):
            sa_vector = np.zeros(nA)
            # Loop through all actions:
            for tuple_info in P[s][policy[s]]:
                # Get tuple
                p, s_, u, _ = tuple_info

                # Sum up possible end states and rewards with given action
                V_new[s] +=  (p* (u + beta * V_old[s_]))

        # Check tolerance
        if la.norm(V_new - V_old) < tol:
            break

        # Reinitialize V_old
        V_old = V_new.copy()

    # Return V_old
    return V_old
    
def test3():
    print(compute_policy_v(P, 4,4, np.array([2,1,2,0])))



# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    
    # Initialize pi_naught
    pi_naught = np.random.choice(nA, nS)

    # Loop maxiter times maximum
    for k in range(0, maxiter + 1, 1):

        # Compute V and pi_new
        V = compute_policy_v(P, nS, nA, pi_naught, beta, tol)
        pi_new = extract_policy(P, nS, nA, V, beta)

        # Check tolerance
        if la.norm(pi_new - pi_naught) < tol:
            break

        # Update pi
        pi_naught = pi_new.copy()


    # Return final V, optimal policy, and number of iterations
    return V, pi_new, k + 1

def test4():
    print(policy_iteration(P, 4,4))


import gym
from gym import wrappers

# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """

    if basic_case == True:

        # Make environment for 4x4 scenario
        env_name = 'FrozenLake-v1'
        env = gym.make(env_name).env

    
    elif basic_case == False:
         
        # Make environment for 8 x8 scenario
        env_name = 'FrozenLake8x8-v1'
        env = gym.make(env_name).env


    # Find number of states and actions
    number_of_states = env.observation_space.n
    number_of_actions = env.env.action_space.n

    # Get the dictionary with all the states and actions
    dictionary_P = env.P

    vi_policy = value_iteration(dictionary_P, number_of_states, number_of_actions)
    vi_total_rewards = 0

    pi_value_func, pi_policy, _ = policy_iteration(dictionary_P, number_of_states, number_of_actions)
    pi_total_rewards = 0

    for m in range(M):

        vi_total_rewards += run_simulation(env, vi_policy)
        pi_total_rewards += run_simulation(env, pi_policy)



    env.close()

    return vi_policy, vi_total_rewards/M, pi_value_func, pi_policy, pi_total_rewards/M





        

    

   
   

# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    
    obs = env.reset()[0]

    done = False
    k = 0
    total_reward = 0
    while done != True:
        obs, reward, done, _ , _= env.step(int(policy[int(obs)]))

        total_reward += reward * beta**k

        k += 1

        if render:
            env.render(mode = 'human')


    env.reset()[0]

    return total_reward

