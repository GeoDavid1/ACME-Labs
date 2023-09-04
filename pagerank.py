# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""
import numpy as np
from numpy import linalg as la
from itertools import combinations as combo

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """

        # Saves dimension of A as a class attribute
        self.n = len(A)

        # If there is a column of zeros, replace it with a column of ones
        A[:, A.sum(axis = 0) == 0] = np.ones((self.n,1))

        # Make columns sum up to 1
        self.Ahat = A / A.sum(axis = 0)

        # Default labels
        if labels is None:
            self.labels = [str(i) for i in range(self.n)]
        else:
            # If labels are given, check if length of labels is dimension of A. If match, set labels, otherwise raise ValueError
            if len(labels) == self.n:
                self.labels = labels
            else:
                raise ValueError("Labels not equal to number of nodes")

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        # Find the PageRank vector p
        p = np.linalg.solve(np.eye(self.n) - epsilon * self.Ahat, (1-epsilon)*np.ones(self.n)/self.n)
        dict = {}

        # Creates a dictionary mapping labels to PageRank values
        for i,x in enumerate(p):
            dict[self.labels[i]] = x
        return dict

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        #Use eigenvector
        vals, vecs = np.linalg.eig(epsilon*self.Ahat + (1-epsilon)*np.ones((self.n,self.n))/self.n)
        p = vecs[:, np.argmax(vals)] / sum(vecs[:, np.argmax(vals)])
        dict = {}

        #Dictionary of PageRank values
        for i, x in enumerate(p):
            dict[self.labels[i]] = x

        return dict
    

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        # Make ones vector
        p = np.ones(self.n)/ self.n
        t = 0

        # Iterates until norm < tol or until t > max iteration
        while t < maxiter:
            oldp = p
            p = epsilon*self.Ahat@p + (1-epsilon)*np.ones(self.n)/self.n
            if(np.linalg.norm(p-oldp, ord = 1) < tol):
                break

        dict = {}

        # Create dictionary
        for i,x in enumerate(p):
            dict[self.labels[i]] = x

        return dict
                
            



def test1():
    A = np.array([[0,0,0,0],[1,0,1,0],[1,0,0,1],[1,0,1,0]])
    dG = DiGraph(A)
    print(dG.linsolve())
    print(dG.eigensolve())
    d = dG.itersolve()
    print(d)


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """

    # Sort the keys
    sorted_dict = {}
    sorted_keys = sorted(d, key = d.get, reverse = True)

    # Make a sorted dictionary
    for w in sorted_keys:
        sorted_dict[w] = d[w]
   
    # Make list of labels fom greatest to least
    needed_list = []
    for x in sorted_dict.keys():
        needed_list.append(x)

    return needed_list

def test1():
    A = np.array([[0,0,0,0],[1,0,1,0],[1,0,0,1],[1,0,1,0]])
    dG = DiGraph(A)
    print(dG.linsolve())
    print(dG.eigensolve())
    d = dG.itersolve()
    print(d)
    print(get_ranks(d))
   


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """

    #Opens file and reads lines
    with open(filename, 'r') as file:
        file = file.readlines()
        # Add webpage id to set
        indices = set()
        for line in file:
            for x in line.strip().split('/'):
                indices.add(x)

        # Converts ids to a list and sorts it
        labels = list(indices)

        # Make adjacency matrix
        A = np.zeros((len(labels), len(labels)))

        # Create a dictionary that maps ID to index in list
        indices = {}
        for i, label in enumerate(labels):
            indices[label] = i
        
        # Make adjacency matrix
        for line in file:
            line = line.strip().split('/')
            index = indices[line.pop(0)]
            for x in line:
                A[indices[x]][index] = 1

        # Create a DiGraph with the adjacency matrix
        graph = DiGraph(A, labels)
        return get_ranks(graph.itersolve(epsilon = epsilon))




# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """

    # Opens file and reads all lines except header
    with open(filename, 'r') as file:
        file = file.readlines()[1:]
        # Gets set of all teams by adding all teams to a set
        indices = set()
        for line in file:
            for x in line.strip().split(','):
                indices.add(x)

        # Converts team set to list and sorts it
        labels = list(indices)
        labels.sort()

        # Initializes adjacency matrix
        A = np.zeros((len(labels), len(labels)))
        indices = {}

        # Creates dictionary mapping team to index
        for i, label in enumerate(labels):
            indices[label] = i

        # Goes through and addds 1 for each game from loser to winner
        for line in file:
            line = line.strip().split(',')
            A[indices[line[0]]][indices[line[1]]] += 1

        # Creates a DiGraph and returns ranked list of team names
        graph = DiGraph(A, labels)
        return get_ranks(graph.itersolve(epsilon = epsilon))
        

import networkx as nx

# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    
    # Open file and read lines
    with open(filename, 'r', encoding = 'utf-8') as file:
        file = file.readlines()
        # Creates a NX DiGraph
        DG = nx.DiGraph()
        map = {}
        for line in file:
            # For each line, splits and sorts actors in reverse order, then makes combinations
            line = line.strip().split('/')
            title = line.pop(0)
            line = line[::-1]
            combos = combo(line,2)
            for c in combos:
                if c in map.keys():
                    map[c] += 1
                else:
                    map[c] = 1
                # Add each combo as an edge
        weighted_edges = [(c[0], c[1], map[c]) for c in map.keys()]
        DG.add_weighted_edges_from(weighted_edges)
        # Return ranked list of actors
        return get_ranks(nx.pagerank(DG, alpha = epsilon))


