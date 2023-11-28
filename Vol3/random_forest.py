"""
Random Forest Lab

Name: David Camacho
Section
Date: 10/31/23
"""
from platform import uname
import os
import graphviz
from uuid import uuid4
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        
        # Initialize variables
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
            
        # Find feature to compare
        feature = sample[self.column]
        
        # Find whether feature is greater or less than and return
        if feature >= self.value:
            return True
        else:
            return False
       
    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    # Initialize arrays and get shape of array
    left = []
    right = []
    m,n = data.shape

    # Find the truth value of every piece of data  
    for datum in data:
        truth_val = question.match(datum)
        if truth_val:
            left.append(datum)
        else:
            right.append(datum)
    
    # Get left and right arrays
    left = np.array(left)
    right = np.array(right)
    
    # Reshape data
    left.reshape(-1,n)
    right.reshape(-1,n)
    
    # Return partitions
    return left, right

# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)

# Problem 2, Problem 6
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
   
    # Find the best gain and best question
    best_gain = 0
    best_question = None
    
    # Find the shape of the data
    m,n = data.shape
    
    # Subset = True
    feature_range = range(n-1)
    if random_subset:
        feature_range = np.random.choice(n-1, int(np.sqrt(n-1)))
    
    # Iterate through columns   
    for col in range(n-1):
        for row in range(m):
            
            # Create the question
            question = Question(column = col, value = data[row,col], feature_names = feature_names)
            
            # Partition the data
            left_part, right_part = partition(data, question)
            
            # Make sure that there are enough samples in each partition
            if len(left_part) < min_samples_leaf or len(right_part) < min_samples_leaf:
                continue
            
            # Find the information gain
            inform_gain = info_gain(data, left_part, right_part)
       
            # Iterate to find the question that creates the best info gain
            if inform_gain > best_gain:
                best_gain = inform_gain
                best_question = question
            
    # Return best gain and best question
    return best_gain, best_question
            
    
# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        
        # Initialize variables
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        
        # Initialze variables
        self.question = question
        self.left = left_branch
        self.right = right_branch

# Prolem 4
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    
    # Initialize variables
    rows, cols = data.shape
    depth = current_depth
    
    # Return leaf if these conditions exist
    if rows < 2 * min_samples_leaf or depth >= max_depth:
        return Leaf(data)
    
    # If not, do this
    else:
        # Get optimal gain and question
        optimal_gain, optimal_question = find_best_split(data, feature_names, min_samples_leaf, random_subset)
        
        # If no optimal gain, return leaf
        if optimal_gain == 0:
            return Leaf(data)
        
    # Get partition
    left_part, right_part = partition(data, optimal_question)
    
    # Increment depth; and build branches
    depth += 1
    left_branch = build_tree(left_part, feature_names, min_samples_leaf, max_depth, depth)
    right_branch = build_tree(right_part, feature_names, min_samples_leaf, max_depth, depth)
    
    # Return decision node
    return Decision_Node(optimal_question, left_branch, right_branch)
        

# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    
    # Determine if the current node (my_tree) is of type Leaf
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key = my_tree.prediction.get)
    
    # If not, recursively call the function on the right side of the tree
    else:
        if my_tree.question.match == True:
            return(predict_tree(sample, my_tree.left))
        else:
            return(predict_tree(sample, my_tree.right))
        
    
def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    
    # Initialize variables
    correct = 0
    
    # Iterate through data
    for data in dataset:
        
        # Get prediction
        label = data[-1]
        predict = predict_tree(data[:-1], my_tree)
        if predict == label:
            correct += 1
    
    # Return proportion of dataset classified correctly
    return correct / len(dataset)
        
   
    
# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    
    # Initialize dictionary of labels
    labels = {}
    
    # Iterate through trees
    for tree in forest:
        
        # Find predicted label
        label = predict_tree(sample, tree)
        if label not in labels:
            labels[label] = 1
        else:
            labels[label] += 1
            
    # Return majority vote of label
    return max(labels, key = labels.get)
        
    
    
    
def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
   
    # Initialize variables
    m, n = dataset.shape
    matches = 0
    
    # Find number of matches
    for datum in dataset:
        label = datum[-1]
        sample = datum[:-1]
        pred_label = predict_forest(sample, forest)
        if label == pred_label:
            matches += 1
    
    # Return proportion of dataset classified correctly     
    return matches / m
    
       
# Problem 7
def prob7():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    
    # Load in data
    start = time.time()
    data = np.loadtxt('parkinsons.csv', delimiter= ',')
    feature_names = np.loadtxt('parkinsons_features.csv', delimiter = ',', dtype =
                               str, comments = None)

    # Random select 130 samples
    np.random.shuffle(data)
    m, n = data.shape
    train_samples = data[:100, 1:]
    test_samples = data[100:130, 1:]
    
    # Create the forest
    forest = [build_tree(train_samples, feature_names, min_samples_leaf = 15,
                         max_depth = 4, random_subset = True) for i in range(5)]
    
    # Predict accuracy
    accuracy = analyze_forest(test_samples, forest)
    finish = time.time()
    tuple_1 = (accuracy, finish - start)
    
    
    # Run the same analysis but for sklearn with non-default parameters
    start = time.time()
    rf = RandomForestClassifier(n_estimators = 5, min_samples_leaf = 15, 
                                max_depth = 4, random_state = 0)
    rf.fit(train_samples[:, :-1], train_samples[:, -1])
    accuracy = rf.score(test_samples[:,:-1], test_samples[:,-1])
    finish = time.time()
    tuple_2 = (accuracy, finish - start)
    
    # Run the same analysis but for sklearn with default parameters
    start = time.time()
    m, n = data.shape
    tr_ratio = int(0.8 * m)
    train_samples = data[:tr_ratio, 1:]
    test_samples = data[tr_ratio:, 1:]
    rf_2 = RandomForestClassifier()
    rf_2.fit(train_samples[:,:-1], train_samples[:, -1])
    accuracy = rf_2.score(test_samples[:, :-1], test_samples[:,-1])
    finish = time.time()
    tuple_3 = (accuracy, finish - start)
    
    return tuple_1, tuple_2, tuple_3



## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if not hasattr(my_tree, "question"):#isinstance(my_tree, leaf_class):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree, filename='Digraph', leaf_class=Leaf):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv',f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)

