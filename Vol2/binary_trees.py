# binary_trees.py
"""Volume 2: Binary Trees.
<Name>
<Class>
<Date>
"""

# These imports are used in BST.draw().
import numpy as np
import random
import time
from anyio import current_effective_deadline
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

               

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """

        # Step Function that Recursively searches a LinkedList

        def _step(current):

            if current is None:
                raise ValueError(str(data) + "is not in the tree.")
            
            if data == current.value:
                return current

            else:
                return _step(current.next)

        
        return _step(self.head)

    

class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """

        # If there is no root, insert the new node as the root

        if (self.root == None):
            self.root = BSTNode(data)

        
        # If not, run a recursive step function

        else:

            def _step(current):

                # Raise Value Error if node is already in thee BST

                if data == current.value:
                    raise ValueError("There is already a node in the tree")

                #  If data is lower than node that we are evaluating, move to the left child
                elif data < current.value:

                    # If there is no left child, insert the node as the left child

                    if(current.left == None):
                        current.left = BSTNode(data)
                        current.left.prev = current

                    # If there is a left child, continue the recursive step function
                    
                    else:
                        return _step(current.left)

                # If data is higher than node we are evaluating, move to the right child
                
                elif data > current.value:

                    # If there is no right child, insert the node as the right child

                    if(current.right == None):
                        current.right = BSTNode(data)
                        current.right.prev = current

                    # If there is a right child, continue the recursive step function

                    else:
                        return _step(current.right)


            # Run the step functions starting at the root

            _step(self.root)

                      

                   
    def __str__(self):

        """ Built-in function from the Lab that prints out the BST nicely"""

        if self.root is None:
            return "[]"
        out, current_level = [], [self.root]
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level

        return "\n".join([str(x) for x in out])

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data"""


        # Raise Value Error if the tree is empty
        
        if(self.root == None):

            raise ValueError ("The tree is empty")

        
        # Find note with requested data
        
        NodeNeeded = self.find(data)

    

        # See which case the requested node is in


        # (1) The target is a leaf node
        
        if (NodeNeeded.left == None and NodeNeeded.right == None):

            # (a) The target is the root

            if (NodeNeeded == self.root):
                self.root = None

            
            # The target is not the root
                
            else:
                
                ParentNode = NodeNeeded.prev
                

            # (b) The target is to the left of its parent
                
                if (NodeNeeded == ParentNode.left):

                    ParentNode.left = None
                    NodeNeeded.prev 

                    return

            # (c) The target is to the right of its parent

                elif (NodeNeeded == ParentNode.right):

                    ParentNode.right = None

                    return

        # (2) The target has two children

        elif (NodeNeeded.left != None and NodeNeeded.right != None):

            # Find its immediate predecessor by stepping to the left of the target and then to the right for as long as possible

            NextNode = NodeNeeded.left
            while(NextNode.right != None):
                NextNode = NextNode.right

            
            # Remove the predecessor, recording its value
            # Then overwrite the value of the target with the predecessor's value

            data = NextNode.value

            self.remove(data)

            NodeNeeded.value = data

            return


            


        # (3) The target has one child

        # (a) The target is the root


        elif(self.root == NodeNeeded):

            # Remove Root and Set Left Child as Root

            if(NodeNeeded.left is not None and NodeNeeded.right is None):

                self.root = NodeNeeded.left
                self.root.prev = None

            # Remove root and Set Right Child as Root
                

            elif (NodeNeeded.left is None and NodeNeeded.right is not None):

                self.root = NodeNeeded.right
                self.root.prev = None
             
            return
        
        # (b) The target is to the left of its parent

        elif(NodeNeeded.value <= NodeNeeded.prev.value):

            # Locate Parent

            Parent = NodeNeeded.prev

            # Locate Child of NodeNeeded

            if (NodeNeeded.left is not None and NodeNeeded.right is None):

                Child = NodeNeeded.left
            
            elif (NodeNeeded.left is None and NodeNeeded.right is not None):

                Child = NodeNeeded.right


            # Set the parent's left attribute to the child and the
            # child's prev attribute to the parent


            Parent.left = Child

            Child.prev = Parent

            return





        # (c) The target is to the right of it parent

        elif(NodeNeeded.value >= NodeNeeded.prev.value):

            # Locate parent

            Parent = NodeNeeded.prev

            # Locate Child of NodeNeeded

            if (NodeNeeded.left != None and NodeNeeded.right == None):

                Child = NodeNeeded.left

            elif (NodeNeeded.left == None and NodeNeeded.right != None):

                Child = NodeNeeded.right

            # Set the parent's right attribute to the child and the
            # child's prev attribute to the parent

            
            Parent.right = Child

            Child.prev = Parent

            return




    
    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.

    """

    # Initialize a SLL, BST, and AVL Tree

    SList = SinglyLinkedList()
    BSTree = BST()
    AVLTree = AVL()

    # Initialize a List of the Lines from 'english.txt"

    ListofLines = []

    # Initialize Load Time Lists

    LoadTimesSList = []
    LoadTimesBSTree = []
    LoadTimesAVLTree = []

    # Initialize Find Time Lists

    FindTimesSList = []
    FindTimesBSTree = []
    FindTimesAVLTree = []

    # Open up 'english.txt', read all the lines, and append each line into
    # ListofLines

    with open('english.txt', 'r') as myfile:
        contents = myfile.readlines()

    for line in contents:
        ListofLines.append(line)


    # Initialize requested number of nodes list

    
    ListN = [8, 16, 32, 64, 128, 256, 512, 1024]

    # Load and Find Times for SLL

    for n in ListN:

        # Get a subset of n random items from the data set ListofLines

        RandomList = []
        tempList = contents

        for i in range(0,n):
            tempChoice = random.choice(tempList)
            RandomList.append(tempChoice)
            tempList.remove(tempChoice)


        # Time how long it takes to load a new SinglyLinkedList of n items


        starttime = time.time()

        for item in RandomList:

            SList.append(item)

        endtime = time.time()

        timeTaken = endtime - starttime

        # Append time to Load Times of SSL

        LoadTimesSList.append(timeTaken)


        # Choose 5 random items from the n random items to form a subset

        SubsetList = []

    
        tempRandomList = RandomList

        for i in range(0,5):
            tempChoice2 = random.choice(tempRandomList)
            SubsetList.append(tempChoice2)
            tempRandomList.remove(tempChoice2)


        # Time how long it takes to find all 5 of these items in a SLL
            
        
        starttime = time.time()

        for item in SubsetList:

            SList.iterative_find(item)

        endtime = time.time()

        timeTaken = endtime - starttime


        # Append time to Find Times of SSL

        FindTimesSList.append(timeTaken)
    
   
    # Load and Find Times for BST
    
    for n in ListN:

        # Get a subset of n random items from the data set ListofLines

        RandomList = []
        tempList = contents

        for i in range(0,n):
            tempChoice = random.choice(tempList)
            RandomList.append(tempChoice)
            tempList.remove(tempChoice)

        
        # Time how long it takes to load a new BST of n items

        starttime = time.time()

        for item in RandomList:

            BSTree.insert(item)

        endtime = time.time()

        timeTaken = endtime - starttime

        # Append time to Load Times of BST

        LoadTimesBSTree.append(timeTaken)


        # Choose 5 random items from the n random items to form a subset

        SubsetList = []

        tempRandomList = RandomList

        for i in range(0,5):
            tempChoice2 = random.choice(tempRandomList)
            SubsetList.append(tempChoice2)
            tempRandomList.remove(tempChoice2)


        # Time how long it takes to find all 5 of these items in a BST

        starttime = time.time()

        for item in SubsetList:
            BSTree.find(item)

        endtime = time.time()

        timeTaken = endtime - starttime


        # Append time to Find Times of BST

        FindTimesBSTree.append(timeTaken)

    

    # Load and Find Times for AVL Trees


    for n in ListN:

        # Get a subset of n random items from the data set ListofLines

        RandomList = []

        tempList = contents

        for i in range(0,n):
            tempChoice = random.choice(tempList)
            RandomList.append(tempChoice)
            tempList.remove(tempChoice)

        
        # Time how long it takes to load a new AVL of n items

        starttime = time.time()

        for item in RandomList:

            AVLTree.insert(item)

        endtime = time.time()

        timeTaken = endtime - starttime


        # Append time to Load Times of AVL


        LoadTimesAVLTree.append(timeTaken)

        # Choose 5 random items from the n random items to form a subset

        SubsetList = []

        tempRandomList = RandomList

        for i in range(0,5):
            tempChoice2 = random.choice(tempRandomList)
            SubsetList.append(tempChoice2)
            tempRandomList.remove(tempChoice2)


        # Time how long it takes to find all 5 of these items in a AVL

        starttime = time.time()

        for item in SubsetList:

            AVLTree.find(item)

        endtime = time.time()

        timeTaken = endtime - starttime


        # Append time to Find Times of AVL

        FindTimesAVLTree.append(timeTaken)


    # Plot Load Times (Build Times) for SLL, BST, and AVL Trees


    ax1 = plt.subplot(121)
    ax1.loglog(ListN, LoadTimesSList, 'k-', label = "LinkedList", base = 2)
    ax1.loglog(ListN, LoadTimesBSTree, 'b-',  label = "BST", base = 2)
    ax1.loglog(ListN, LoadTimesAVLTree, 'r-', label = "AVL Trees", base = 2)
    plt.title("Build Times")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time (Seconds)")
    plt.legend(loc = "upper left")


    # Plot Search Times for SLL, BST, and AVL Trees

    ax2 = plt.subplot(122)
    ax2.loglog(ListN, FindTimesSList, 'k-', label = "LinkedList", base = 2)
    ax2.loglog(ListN, FindTimesBSTree, 'b-',  label = "BST", base = 2)
    ax2.loglog(ListN, FindTimesAVLTree, 'r-', label = "AVL Trees", base = 2)
    plt.title("Search Times")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time (Seconds)")
    plt.legend(loc = "upper left")


    # Show subplots

    plt.show()




   







        





    


    




    
