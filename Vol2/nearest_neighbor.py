# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree

from scipy.stats import mode


# Problem 1
def exhaustive_search(X, z):

    # Initialize Variables

    m,n = np.shape(X)
    MinNorm = np.inf
    indexMinNorm = -1

    # Find the list of Norms
    NormNeighbors = la.norm(X - z, axis = 1)

    # Find the minimum norm

    for i in range (0, len(NormNeighbors)):

        if NormNeighbors[i] < MinNorm:

            MinNorm = NormNeighbors[i]
            indexMinNorm = i

    # Find the Nearest Neighbor
    NearestNeighbor = X[indexMinNorm, :]

    # Return nearest neighbor and its Euclidean distance
    return NearestNeighbor, MinNorm

# Problem 2: Write a KDTNode class.
class KDTNode:

    def __init__(self, x):

        # Raise a TypeError is thee type is not an np.ndarray
        if type(x) != np.ndarray:

            raise TypeError ("Type is wrong")

        # Initialize attributes
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None
        

# Problems 3 and 4
class KDT:
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)


    def __str__(self):
        
        """String representation: a hierarchical list of nodes and their axes.
            Example: 'KDT(k=2)
            [5,5] [5 5] pivot = 0
            / \ [3 2] pivot = 1
            [3,2] [8,4] [8 4] pivot = 1
            \ \ [2 6] pivot = 0
            [2,6] [7,5] [7 5] pivot = 0'
            
            
            """
        if self.root is None:
            return "Empty KDT"

        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
       
        # If the tree is nonempty, create a new KDT Node containing data and set its pivot to 0
        if(self.root == None):

            newKDTNode = KDTNode(data)
            newKDTNode.pivot = 0

            # Set attributes

            self.root = newKDTNode
            self.k = len(data)

        
        else:
            # Raise ValueError is data to be inserted is not in Rk
            if(len(data)!= self.k):

                raise ValueError("Data needs to have length k")

            newKDTNode = KDTNode(data)


            # Recurrence function to find where to insert the new node

            def _step(current):

                # Raise ValueError is the node is not in the tree
                if current is None:

                    raise ValueError(str(data) + " is not in the tree")

                # Raise ValueError if theere is a duplicate in the tree
                elif np.allclose(data, current.value):

                    raise ValueError ("Duplicate found: Not allowed")

                # If data is less than the current value, do this:
                elif data[current.pivot] < current.value[current.pivot]:

                    # If there is no node, insert here
                    if (current.left == None):

                        current.left = newKDTNode
                        
                        # Set new pivot
                        if (current.pivot != self.k - 1):

                            newKDTNode.pivot = current.pivot + 1

                        else:

                            newKDTNode.pivot = 0

                    # If not, continue the recursion
                    else:

                        return _step(current.left)

                # If data is more than the current value, do this:
                elif data[current.pivot] > current.value[current.pivot]:

                    # If there is no node, insert here
                    if (current.right == None):

                        current.right = newKDTNode

                        # Set new pivot
                        if (current.pivot != self.k -1):

                            newKDTNode.pivot = current.pivot + 1

                        else:

                            newKDTNode.pivot = 0

                    # If not, continue the recursion
                    else:

                        return _step(current.right)

            # Run the recursion method on the root

            return _step(self.root)


    # Problem 4
    def query(self, z):

        def KDSearch(current, nearest, d):

            # Base Case: Dead End
            if current == None:
                return nearest, d

            x = current.value
            i = current.pivot

            # Check if current is closer to z than nearest
            if la.norm(x - z) < d:

                nearest = current
                d = la.norm(x -z)

            # Search to the left
            if z[i] < x[i]:

                nearest, d = KDSearch(current.left, nearest, d)

                # Search to the right if needed
                if (z[i] + d >= x[i]):

                    nearest, d = KDSearch(current.right, nearest, d)

            # Search to the right
            else:

                nearest, d = KDSearch(current.right, nearest, d)

                # Search to the left if needed
                if(z[i] - d <= x[i]):

                    nearest, d = KDSearch(current.left, nearest, d)

            # Return desired values
            return nearest, d

        # Run the KDSearch on the root to get the Nearest Neighbor Search
        node, d = KDSearch(self.root, self.root, la.norm(self.root.value - z))
        return node.value, d

                

    

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """

    def __init__(self, n_neighbors):

        # Initialize numNeighbors
        self.numNeighbors = n_neighbors

    
    def fit(self, X, y):

        # Make a KDTree of the data; make attributes
        neededTree = KDTree(X)
        self.tree = neededTree
        self.labels = y

    
    def predict(self, z):

        # Query the KDT Tree to get the closest neighbors
        distances, indices = self.tree.query(z, k = self.numNeighbors)

        # Find mode and return mode
        occurs_most = mode(self.labels[indices])[0][0]
        return occurs_most


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):

    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """

    # Load data
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float64)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float64)
    y_test = data["y_test"]

    # Load classifier and fit the data to the classifier
    neededKNC = KNeighborsClassifier(n_neighbors)
    neededKNC.fit(X_train, y_train)

    # Find classification accuracy
    correct = 0
    incorrect = 0

    # Match data to y_test and find accuracy
    for i in range(len(X_test)):
        y_pred = neededKNC.predict(X_test[i])

        if y_pred == y_test[i]:

            correct += 1

    # Return classification accuracy
    return correct/ len(X_test)

        




        



    




