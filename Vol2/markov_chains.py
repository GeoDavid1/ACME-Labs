# markov_chains.py
"""Volume 2: Markov Chains.
<Name>
<Class>
<Date>
"""

from importlib.machinery import WindowsRegistryFinder
import numpy as np
from scipy import linalg as la
from sklearn.preprocessing import normalize


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:

    self.matrix: Transition Matrix
    self.labels: Matrix Labels
    self.attr_dict: a dictionary that maps labels to row/column indices
    """
    # Problem 1
    def __init__(self, A, states=None):

        # Check to see if A is column stochastic
        length, width = A.shape
        if not np.allclose(np.ones(length), A.sum(axis = 0)):

            raise ValueError("Matrix A is not column stochastic")

        # Check to see if A is square
        m,n = np.shape(A)
        if m !=n:
            raise ValueError("Matrix A is not square")

        # Create labels
        self.matrix = A
        if states == None:
            self.labels = [i for i in range(np.shape(A)[1])]
        elif states != None:
            self.labels = states

        # Map the state labels to the row/column index with a dictionary
        attributes = {}
        for position, label in enumerate(self.labels):
            attributes[label] = position

        # Save dictionary as attribute
        self.attr_dict = attributes


    
    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """

        # Raise error if not valid state
        if state not in self.attr_dict:
            raise ValueError(f"{state} is not a valid state")

        # Find column of A
        tomorrow_position = self.attr_dict[state]
        tomorrow_column = self.matrix[:, tomorrow_position]/sum(self.matrix[:, tomorrow_position])

        # Make corresponding categorical distribution to choose to state to transition to
        transition = np.random.multinomial(1, tomorrow_column)
        next_state = np.argmax(transition)

        # Return label to new state
        return list(self.attr_dict.keys())[list(self.attr_dict.values()).index(next_state)]

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """

        # Initialize list
        transition_list = []

        # Starting at the specified state, use transition() to transition N-1 times
        for i in range(N):

            state_label = self.transition(start)

            # Record the state label at each step
            transition_list.append(state_label)
            start = state_label

        # Return state labels
        return transition_list
            

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        # Initialize current state
        current_state = start

        # Start path list
        path_list = [start]

        # Transition from state to state until arriving at specified end state
        # Record the state label at each step
        while current_state != stop:
            current_state = self.transition(current_state)
            path_list.append(current_state)

        # Return the list of state labels, including the initial and final states
        return path_list


    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        # Generate random state distribution vector
        length = np.shape(self.matrix)[0]
        state_dist = np.random.dirichlet(np.ones(length), size = 1)

        # Calculate x_k
        x_k = state_dist.reshape((length,))

        # Transition for maxiter number of times
        for k in range(maxiter):
            x_k1 = self.matrix@x_k

            if np.linalg.norm((x_k - x_k1), ord = 1) < tol:
                return x_k1
            x_k = x_k1

        # Raise ValueError if A^k does not converge
        raise ValueError("Did not reach steady point, so A^k does not converge")



class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        self.matrix: Transition Matrix
        self.labels: Matrix Labels
        self.attr_dict: a dictionary that maps labels to row/column indices
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """

        # Open file
        with open(filename, "r") as my_file:

            line = my_file.read()
            words = line.split()
            sentences = line.split('\n')

        # Get Set of Unique Words in the training set (the state labels)
        words_set = set(words)

        # Add "$tart" and "$top"
        words_set.add("$tart")
        words_set.add("$top")
        words_list = list(words_set)

        # Initialize An Appropriately Sized Square Array of Zeros to Be The Transition Matrix
        transition_matrix = np.zeros((len(words_set), len(words_set)))

        for sent in sentences:
            # Split the sentence into a list of words
            split_sentence = sent.split()

            # Prepend "$tart" and "$top" to list of words
            split_sentence.insert(0, "$tart")
            split_sentence += ["$top"]

            for i in range(len(split_sentence) -1):
                
                # Add 1 to the entry of the transition matrix that corresponds to transitioning from
                # state x to state y
                word_1 = split_sentence[i]
                word_2 = split_sentence[i+1]
                index_2 = words_list.index(word_1)
                index_1 = words_list.index(word_2)
                transition_matrix[index_1, index_2] += 1

        # Make sure the stop state transitions to itself
        index = words_list.index("$top")
        transition_matrix[index, index] += 1

        # Normalize each column by dividing by the column sums
        x, _ = transition_matrix.shape
        for i in range(x):
            col_sum = np.sum(transition_matrix[:,1])
            if col_sum == 0:
                continue
            else:
                transition_matrix[:,i] /= col_sum

        # Save attributes
        self.matrix = transition_matrix
        self.labels = words_list
        attributes = {}
        for position, label in enumerate(self.labels):
            attributes[label] = position
        self.attr_dict = attributes


    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """

        # Generate a random sentence
        sentence = self.path("$tart", "$top")
        
        # Remove "$top" and "$tart"
        sentence.remove("$top")
        sentence.remove("$tart")

        # Return sentence
        return " ".join(sentence)

        
