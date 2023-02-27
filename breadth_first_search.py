# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Name>
<Class>
<Date>
"""

import collections
import networkx as nx
import matplotlib.pyplot as plt
import statistics

from sklearn.neighbors import NeighborhoodComponentsAnalysis


# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):

        # Use get function to add empty set if not in graph
       self.d[n] = self.d.get(n, set())


    # Problem 1
    def add_edge(self, u, v):

        # Check first if in graph, and add if not
        self.add_node(u)
        self.add_node(v)

        # Add necessary edges
        self.d[u].add(v)
        self.d[u].add(u)



    # Problem 1
    def remove_node(self, n):

        #Remove key itself
        self.d.pop(n)

        #Iterate through dictionary
        for k in self.d.keys():

            # Remove if edge to popped node exists
            try:
                self.d[k].remove(n)

            #Ignore if an error is raised
            except:
                pass
            

    # Problem 1
    def remove_edge(self, u, v):

        # Remove of edge between given nodes
        self.d[u].remove(v)
        self.d[v].remove(u)


    # Problem 2
    def traverse(self, source):

        # Initialize V,Q,M
        V = []
        Q = collections.deque()
        M = set()

        # Append source node to Q and M
        if source in self.d.keys():

            Q.append(source)
            M.add(source)

        # Traverse the graph with a BFS until all nodes have been visited
            while (len(Q) != 0):

                # Pop node off Q, and append it to V
                currentNode = Q.popleft()
                V.append(currentNode)

                # Find neighbors
                Neighbors = self.d[currentNode]

                # Add the neighbors of the current node that are not in M to Q and M
                for neighbor in Neighbors:
                    if neighbor not in M:
                        Q.append(neighbor)
                        M.add(neighbor)

        # If the source node is not in the graph, raise a KeyError
        else:
            raise KeyError ("Source node is not in the graph")
        
        # Return desired value
        return V


    # Problem 3
    def shortest_path(self, source, target):
        
        # Initialize dictionaries and lists
        NodesVisited = []
        Q = collections.deque()
        M = set()
        BFSDict = dict()

        # Run the shortest_path function only if the source and target are nodes
        if source in self.d.keys() and target in self.d.keys():

            # Append to Q and add to M

            Q.append(source)
            M.add(source)

            # Run traverse function until you visit the target

            while target not in NodesVisited:

                currentNode = Q.popleft()
                NodesVisited.append(currentNode)

                # Find neighbors
                Neighbors = self.d[currentNode]

                # Create the key-value pair mapping the visited node to the visiting node
                for neighbor in Neighbors:

                    if neighbor not in M:
                        BFSDict.update({neighbor: currentNode})
                        Q.append(neighbor)
                        M.add(neighbor)

            # Make a list containing the node values in the shortest path
            # from the source to the target (including endpoints)
            EndList = []
            EndList.append(target)
            while source not in EndList:

                EndList.insert(0, BFSDict[target])
                target = BFSDict[target]

        # Raise KeyError if one of the nodes does not exist
        else:
            raise KeyError("One of the input nodes are not in the graph")

        # Return desired list
        return EndList



# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):

        # Open file
        with open(filename, 'r') as myfile:
            contents = myfile.readlines()

        # Initialize sets and nx.Graph
        MovieTitles = set()
        ActorNames = set()
        Network = nx.Graph()

        # Add the title to the set of movies 
        for content in contents:
            tempList = content.strip().split('/')
            MovieTitle = tempList[0]
            ActorsInMovie = tempList[1:]
            MovieTitles.add(MovieTitle)

        # Add the cast members to the set of actors
            for actor in tempList[1:]:
                ActorNames.add(actor)

        # Add an edge to the graph between the movie and each cast member
            for actor in ActorsInMovie:
                Network.add_edge(MovieTitle, actor)

        # Store attributes
        self.MT = MovieTitles
        self.AN = ActorNames
        self.Net = Network



    # Problem 5
    def path_to_actor(self, source, target):
        
        # Return shortest path and number of steps from source to target, excluding movies
         
         short_path = nx.shortest_path(self.Net, source, target)

         return short_path, len(short_path)//2
        
        

    # Problem 6 
    def average_number(self, target):

        # Initialize Dictionaries
        ShortestPathLengths = dict(nx.single_target_shortest_path_length(self.Net, target))
        MoviePathLengths = dict()

        # Get rid of all the movies from the dictionaries
        for key in ShortestPathLengths:
            if ShortestPathLengths[key] % 2 == 0:
                MoviePathLengths[key] = ShortestPathLengths[key] /2


        # Initialize the x-axis for the histogram; find mean size of path
        x = MoviePathLengths.values()
        MeanPath = statistics.mean(list(x))

        # Plot histogram
        plt.hist(x, bins = [i - 0.5 for i in range(8)])
        plt.title("Distribution of Path Lengths")
        plt.xlabel("Path Length from Target")
        plt.ylabel("Number of Actors")
        plt.show()

        # Return average path length
        return MeanPath










       


