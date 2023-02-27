# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Name>
<Class>
<Date>
"""

import math

class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color(str): the color of the backpack.
        max_size(int): number of items the backpack can hold
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name, color and maximum size of the backpack

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size(int): number of items the backpack can hold
        """
        self.name = name                #Initialize values of the backpack
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):

        """Add an item to the backpack's list of contents if there is room in the backpack.
        If there is no room, do not add the item to the backpack.

        Parameters:
            item(str): the name of the item that is going to be added to the backpack."""

        if(len(self.contents) < self.max_size): # If there is room
        
            self.contents.append(item) 

        else:                                   # If there is no room
            print("No Room!")

    def dump(self):

        """Clear the contents of the backpack """

        self.contents.clear()                       # Clear backpacks

    def take(self, item):
        """Remove an item from the backpack's list of contents."""

        """Parameters:
            item(str): the name of the item that is going to be removed from the backpack."""

        self.contents.remove(item)                  #Remove item


    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):

        """Add the number of contents of each Backpack."""

        return len(self.contents) + len(other.contents)                # Add number of contents of a backpack

    def __lt__(self, other):

        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)                 #Return if backpack has fewer items than another backpack
    
    def __eq__(self, other):

        """ Compare two backpacks. If 'self' has the same number of contents than 'other',
        return True. Otherwise, return False"""

        return len(self.contents) == len(other.contents) and self.name == other.name and self.color == other.color    #Return if the backpacks are the same

    def __str__(self):

        "Print out the desired string listing the owner, color, size, max size, and contents"

        #Print Sting that the Lab desires

        String_Needed = "Owner:" + '\t\t' + self.name + '\n' + "Color:" + '\t\t' + self.color + '\n' + "Size:" + '\t\t'                         
        String_Needed +=  str(len(self.contents)) + '\n' + 'Max Size:' + '\t' + str(self.max_size) + '\n' + 'Contents:' +'\t' + str(self.contents)
        
        return String_Needed



# Test Function for Backpack
def test_backpack():
    testpack = Backpack("Barry", "black")
    other = Backpack("Barry", "green")
    testpack.put("crayons")
    testpack.put("pencils")
    print(testpack.contents)
    testpack.take("crayons")
    print(testpack.contents)
    testpack.dump()
    print(testpack.contents)
    print(testpack == other)
    print(str(testpack))
    




# An example of inheritance. You are not required to modify this class.

class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)         #Initialize Backpack 
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")             #If tied, print "I'm closed"
        else:
            Backpack.put(self, item)         #If untied, put an iteem in the backpack    

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")            #If tied, print "I'm closed"
        else:
            Backpack.take(self, item)       #If untied, take away item

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)      #Return length of contents array



# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.

class Jetpack(Backpack):
    "A Jetpack object class. Inherits from the Backpack class. A jetpack also contains a fuel tank"


    """Attributes:
            name (str): the name of the jetpack's owner.
            color (str): the color of the jetpack.
            max_size(int): the maximum number of items that can fit inside.
            contents (list): the contents of the backpack
            amount_fuel(int): the amount of fuel in the fuel tank
    """
        
    def __init__(self, name, color, max_size = 2, amount_fuel = 10):

        """Use the Backpack constructor to initialize the name, color, and max_size attributes.
        A jetpack only holds 2 items by default. Additionally, a jetpack holds 10 units of fuel"""


        """ Parameters:
                name (str): the name of the jetpack's owner
                color (str): the color of the jetpack.
                max_size(int): the maximum number of items that can fit inside
                amount_fuel(int): the amount of fuel in the fuel tank
        
        """

        Backpack.__init__(self, name, color, max_size)      # Initialize backpack
        self.amount_fuel = amount_fuel                      # Set fuel amount

    def fly(self, fuel_to_burn):

        """ If the jetpack has enough fuel, have the jetpack fly and decrement the fuel burnt from the total fuel"""

        """ Parameters:
                fuel_to_burn (float): amount of fuel used"""

        # If there is enough fuel, do this

        if (fuel_to_burn <= self.amount_fuel):
            self.amount_fuel -= fuel_to_burn

        # Else, print("Not enough fuel!")

        else:
            print("Not enough fuel!")

    def dump(self):

        """ Clear the contents of the jetpack and set the amount of fuel to 0 """

        self.contents.clear()               #Clear contents
        self.amount_fuel = 0                #Set Fuel = 0


# Test function for Jetpack

def test_jetpack():
    testpack = Jetpack("Barry", "black")
    other = Backpack("Barry", "green")
    testpack.put("crayons")
    testpack.put("pencils")
    print(testpack.contents)
    testpack.take("crayons")
    print(testpack.contents)
    print(testpack.contents)
    print(testpack == other)
    print(str(testpack))
    print(str(testpack.amount_fuel))
    testpack.fly(5)
    print(str(testpack.amount_fuel))

# Problem 4: Write a 'ComplexNumber' class.


class ComplexNumber:

    "A ComplexNumber object class. Has a real portion and an imaginary portion"

    """ Attributes:
        real(float): the real portion of the complex number
        imag(float): the imaginary portion of the complex number"""

    def __init__(self, real, imag):

        """ Set the real and complex parts of the complex number"""

        self.real = real                                #Initialize Real Part
        self.imag = imag                                #Initialize Imaginary Part

    def conjugate(self):

        """Find the complex conjugate of the complex number"""

        ComplexConjugate = ComplexNumber(self.real, -1*(self.imag))  #Find Complex Conjugate
        return ComplexConjugate
    
    def __str__(self):

        """Print out a + bi as (a+bj) for b >= 0 and (a-bj) for b < 0 """

        # Make the following string if self.imag >= 0

        if(self.imag >= 0):

            String_To_Print = "(" + str(self.real) + "+" +  str(self.imag) + "j)"

        # Make the following string if self.imag <= 0
        else:

            String_To_Print = "(" + str(self.real) + str(self.imag) + "j)"

        # Return the string

        return str(String_To_Print)

    def __abs__(self):

        """ Find the magnitude of the complex number (sqrt(a**2 + b**2))"""

        Magnitude = math.sqrt(((self.real)**2) + (self.imag)**2)    #Calculate the Magnitude
    
        return Magnitude

    def __eq__(self, other):

        """ Check to see if two ComplexNumber objects are equal to each other (real and imaginary parts are equal)"""



        # If the two ComplexNumber objects are equal, do this

        if((self.real == other.real) and (self.imag == other.imag)):    

            return True

        #If not, do this

        else:

            return False

    def __add__(self, other):

        """ Add two complex numbers together """

        sum_real = self.real + other.real              
        sum_imag = self.imag + other.imag       
        ComplexSum = ComplexNumber(sum_real, sum_imag)    # Make a new complex number of the sum
        return ComplexSum

    def __sub__(self,other):

        """Subtract two complex numbers together"""

        dif_real = self.real - other.real
        dif_imag = self.imag - other.imag
        ComplexDif = ComplexNumber(dif_real, dif_imag)   # Make a new complex number of the difference
        return ComplexDif


    def __mul__(self,other):

        """Multiply two complex numbers together (FOIL method)"""

        mul_real = self.real * other.real - (self.imag * other.imag)
        mul_imag = self.real * other.imag + (self.imag * other.real)
        ComplexMul = ComplexNumber(mul_real, mul_imag)                  # Make a new complex number of the product
        return ComplexMul

    def __truediv__(self, other):

        """Divide two complex numbers together"""

        truediv_real = ((self.real * other.real) + (self.imag * other.imag))/(((other.real)**2) + ((other.imag)**2))
        truediv_imag = ((self.imag * other.real) - (self.real * other.imag))/(((other.real)**2) + ((other.imag)**2))
        ComplexTrueDiv = ComplexNumber(truediv_real, truediv_imag)          # Make a new complex number of the quotient
        return ComplexTrueDiv

# Test Function for Complex Number

def test_ComplexNumber(a,b):

    aComplexNumber = ComplexNumber(a,b)
    other = ComplexNumber(5,12)
    print(aComplexNumber.conjugate())
    print(aComplexNumber)
    print(abs(aComplexNumber))
    print(aComplexNumber == other)
    print(aComplexNumber + other)
    print(aComplexNumber - other)
    print(aComplexNumber * other)
    print(aComplexNumber / other)


   


