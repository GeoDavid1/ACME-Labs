# linked_lists.py
"""Volume 2: Linked Lists.
<Name>
<Class>
<Date>
"""


# Problem 1


class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        self.value = data
        if (type(data) != int and type(data) != float and type(data) != str):

            raise TypeError("Data needs to be of type int, float, or string")


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.size += 1

        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            self.size += 1

    # Problem 2
    def find(self, data):
        
        """Find and return the first node in the list containing that data"""

        
        # If the list is empty, raise an error

        if (self.head == None):

            raise ValueError("List is empty")

        # Run a while loop until you find the node that contains the data

        TempElement = self.head 

        while(TempElement.value != self.tail.value):

            if(TempElement.value == data):
                return TempElement
                    
                    
            # Keep going

            else:
                TempElement = TempElement.next


        # Stop the while loop when you reach the tail end of the list


        if(TempElement.value == self.tail.value):
            
            if(TempElement.value == data):
                return TempElement


            # We have reached the end, and still have not found the desired node, thus raise an error

            else:

                raise ValueError("No such node exists")


    # Problem 2
    def get(self, i):
        """Return the i-th node in the list."""



        # Raise error if index is negative

        if (i < 0):
            raise IndexError("Index is negative")

        # Raise error if index is out of bounds

        if (i > self.size):
            raise IndexError("Index is greater than or equal to the current number of nodes")

        # Starting from the head, go down the linked list i times to get desired node

        TempElement = self.head

        for counter in range(0, i):
            TempElement = TempElement.next


        # Return desired node

        return TempElement

        
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def __len__(self):

        """Return the number of nodes in the list."""

        return self.size
        

    # Problem 3
    def __str__(self):

        # Start desired string

        string = "["

        # Starting from the beginning, append to string each element in the LinkedList

        TempElement = self.head

        if TempElement is not None:
            while TempElement.next is not None:

                # Add desired aesethics and the next element

                string += repr(TempElement.value) + ", "
                TempElement = TempElement.next

            # On the last element, do not add the comma

            string += repr(TempElement.value)

        # Return desired string

        return string + "]"

    




    

    # Problem 4
    def remove(self, data):

        # First of all, find the data

        NodeNeeded = self.find(data)

        # If the desired node is the only node, do this:

        if(NodeNeeded == self.head and NodeNeeded == self.tail):

            self.head = None
            self.tail = None
            self.size -= 1

        # If the desired node is 1st node, do this:

        elif(NodeNeeded == self.head):

            self.head = NodeNeeded.next
            NodeNeeded.next.prev = None
            self.size -= 1

        # If the desired node is the last node, do this:

        elif(NodeNeeded == self.tail):

            self.tail = NodeNeeded.prev
            NodeNeeded.prev.next = None
            self.size -= 1

        # If the desired node is none of the above, do this:

        elif(NodeNeeded != self.head and NodeNeeded != self.tail):

            NodeNeeded.prev.next = NodeNeeded.next
            NodeNeeded.next.prev = NodeNeeded.prev
            self.size -= 1


    # Problem 5
    def insert(self, index, data):

        # If index is greater than length or negative, raise error

    
        if (index > len(self) or (index < 0)):
            raise IndexError("Index is out of range")

        # Create a new Node for this data point

        newNode = LinkedListNode(data)
    

        # If the LinkedList is of length 0, or if the index is at the end, simply append the Node

        if len(self) == 0 or index == len(self):

            self.append(data)

        # If index = 0, make sure to not reference the "previous" node

        elif index == 0:

            next = self.get(0)   #Get the current 0th Node
            next.prev = newNode   #Do the switch of pointers to add the Node to the beginning
            newNode.next = next
            self.head = newNode
            self.size += 1         # Size of LinkedList is increased by 1

        
        # If index is not 0, make sure to reference the previous node

        else:

            next = self.get(index)  #Get the current ith Node
            previous = next.prev    # Get the Node before
    

            next.prev = newNode     # Do the switch of pointers to add the Node to the desired index
            previous.next = newNode
            newNode.prev = previous
            newNode.next = next
            self.size += 1          # Size of LinkedList is increased by 1

        

# Problem 6: Deque class.

# Make a subclass of LinkedList called Deque

class Deque (LinkedList):

    #Initialize the constructor by inheriting everything from the LinkedList class

    def __init__(self):

        LinkedList.__init__(self)
        self.size = 0  # To make sure the size parameter works

    
    # Remove Node from tail

    def pop(self):

        #If Deque has no elements, return ValueError

        if(self.size == 0):

            raise ValueError("List is empty")

        #If Deque has one element, get rid of the head and tail (the one element)

        elif(self.size == 1):

            LastNode = self.tail
            self.head = None
            self.tail = None
            self.size -= 1   # Size of Deque is decreased by 1

            # Return removed value

            return LastNode.value

        else:

            #If Deque has more than one element, get rid of the tail
        
            LastNode = self.tail
            self.tail = LastNode.prev
            LastNode.prev.next = None
            self.size -=1           # Size of Deque is decreased by 1

            # Return removed value
        
            return LastNode.value

    
    # Remove Node from head

    def popleft(self):

        #If Deque has no elements, return ValueError

        if(self.size == 0):

            raise ValueError("List is empty")

        #If Deque has one element, get rid of the head and tail (the one element)

        elif(self.size == 1):

            FirstNode = self.head
            self.head = None
            self.tail = None
            self.size -= 1       # Size of Deque is decreased by 1

            # Return removed value

            return FirstNode.value


        else:

            #If Deque has more than one element, get rid of the head

            FirstNode = self.head
            self.head = FirstNode.next
            FirstNode.next.prev = None
            self.size -= 1       # Size of Deque is decreased by 1

            # Return removed value

            return FirstNode.value



    # Append Node to the 0th index of the Deque

    def appendleft(self, data):

        LinkedList.insert(self, 0, data)    #Inherited function from LinkedList


    # Disable remove function

    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")

    # Disable insert function

    def insert(*args, **kwargs):
        raise NotImplementedError("Use append() or appendleft()")

    
    

    

# Problem 7
def prob7(infile, outfile):

    # Create a Deque to store the contents of infile

    NeededDeque = Deque()

    # Open infile, read the lines of infile, and close infile
    
    myfile = open(infile, 'r')
    contents = myfile.readlines()
    myfile.close()

    # Add each line of the infile as a Node in NeededDeque

    for line in contents:
        NeededDeque.append(line)

    # Write each line (in opposite order) to the outfile

    with open(outfile, 'w') as outfile:
        for i in range(0, len(NeededDeque)):
            outfile.write(NeededDeque.pop())


    # Close outfile

    outfile.close()







   



    






 
