# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>
<Class>
<Date>
"""

from csv import Dialect
from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    # Step 1


    #Input needed (a 3-digit number)
    
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")

    # Check to see if the input is a 3-digit number

    if int(step_1) < 100 or int(step_1) > 999:

        raise ValueError("You need to choose a three digit number")


    # Find first and last digits of the number

    Step1FirstDigit = step_1[0]
    Step1LastDigit = step_1[2]


    # Calculate to see if the first and last digits differ by more than 2, if not raise an error

    if abs(int(Step1FirstDigit) - int(Step1LastDigit)) < 2:
        raise ValueError("The first and Last Digits need to differ by more than 2")

    

    # Step 2
    # Input needed (the reverse of the number)

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")

    
    # Check to see if the second input is the reverse of the first number, if not raise error

    if(int(step_2[::-1]) != int(step_1)):
        raise ValueError("The second number is not the reverse of the first number.")

    

    # Step 3
    # Input needed (the positive difference of these 2 numbers)

    step_3 = input("Enter the positive difference of these numbers: ")

    # Calculate Difference

    Difference = abs(int(step_1) - int(step_2))

    # See if Difference is the actual difference, if not raise eror

    if (int(step_3) != Difference): 
        raise ValueError("The third number is not the positive difference of the first two numbers")


    #Step 4
    # Input needed (the reverse of Step 3)

    step_4 = input("Enter the reverse of the previous result: ")

    # See if the fourth input is the reverse of the third input, if not raise error

    if (int(step_3[::-1]) != int(step_4)):
        raise ValueError("The fourth number is not the reverse of the third number")


    # Print requested string


    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    
    """If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk."""
    
    #Initialize variables

    walk = 0
    directions = [1, -1]

    #Iterate the random_walk as long as there is no KeyboardInterrupt

    try:
        for i in range(int(max_iters)):
            walk += choice(directions)

    #If ctrl+c is entered, raise KeyboardInterupt Exception

    except KeyboardInterrupt:

        print("Process interrupted at iteration " + str(i))

    #If ctrl+c is not entered, print "Process Completed")

    else:
        print("Process completed")
    
    #Return ending location

    return walk








# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
    # Problem 3

    # Constructor of File Reader

    def __init__(self, filename):

        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """

        #Start with the file not being opening

        file_Opened = False

        #As long as we have not opened the file, run this:

        while(file_Opened == False):

            # If the file is valid, change file_Opened to True and be done

            try: 
                with open(filename, 'r') as myfile:
                    self.contents = myfile.read()

                    self.filename = filename
                    file_Opened = True
        
            # If the file is invalid, request for a new (hopefully valid) file name

            except Exception as e:

                filename = input ("Please enter a valid file name:")
        
 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):

        # If mode is not correct (w,x,a), raise ValueError

        if mode != 'w' or 'x' or 'a':

            raise ValueError("Mode needs to be 'w', 'x', or 'a'")


    

    def uniform(self, outfile, mode='w', case='upper'):

        #Open file

        with open(outfile, mode) as outfile:

            # If case is upper, write the file in all uppercase
            if case == 'upper':
                stringNeeded = self.contents.upper()
                outfile.write(stringNeeded)

            # If case is lower, write the file in all lowercase

            elif case == 'lower':
                stringNeeded = self.contents.lower()
                outfile.write(stringNeeded)

            # If neither, then there is a problem, so raise ValueError

            else:
                raise ValueError


    def reverse(self, outfile, mode='w', unit='line'):

        #Open file

        with open(outfile, mode) as outfile:

            # Split lines

           lines = [line.split() for line in self.contents.strip().split('\n')]

           #If unit is word, reverse the ordering of the words in each lines, but write the lines
           # in the same order as the original file.

           # Create desired data order

           if unit == 'word':

                data = []
                for Line in lines:
                    Line = Line[::-1]
                    data.append(Line)


            # Read in the data into outfile

                for line in data:
                    l = ""
                    for word in line:
                        l += word + " "
                    outfile.write(l + '\n')

            # If unit is line, reverse the ordering of the lines, but do not change the ordering
            # of the words on each individual line.

            # Create desired data order
            
           elif unit == 'line':
            
                data = [0] * len(lines)
                for i in range(1, len(lines) + 1):
                    data[i-1] = lines[-i]

            # Read in the data into outfile
                
                for line in data:
                    l = ""
                    for word in line:
                        l += word + " "
                    outfile.write(l + '\n')

            # If unit is not word or line, there is a problem, so raise a ValueError
          
           else:
                raise ValueError("Please have mode be word or line")



    def transpose(self, outfile, mode='w'):

        # Open file

        with open(outfile, mode) as outfile:

            # Split lines

            lines = [line.split() for line in self.contents.strip().split('\n')]
            
            # Calculate max length of iteration

            max_len = max([len(line) for line in lines])

            # Write the "transpose" of the file into the outfile

            for i in range(0, max_len):
                for line in lines:
                    if i < len(line):
                        outfile.write(str(line[i]) + " ")
                outfile.write('\n')
            

    def __str__(self):

        # Split lines

        lines = [line.split() for line in self.contents.strip().split('\n')]

        # Create requested string of information

        String_Needed = "Source file:" + '\t\t' + self.filename + '\n'
        String_Needed += "Total characters:" + '\t' + str(len(self.contents)) + '\n'
        String_Needed += "Alphabetic characters:" + '\t' + str(sum([s.isalpha() for s in self.contents])) + '\n'
        String_Needed += "Numerical characters:" + '\t' + str(sum([s.isdigit() for s in self.contents])) + '\n'
        String_Needed += "Whitespace characters:" + '\t' + str(sum([s.isspace() for s in self.contents])) + '\n'
        String_Needed += "Number of lines:" + '\t' + str(len(lines))


        # Return requested string of information

        return String_Needed


