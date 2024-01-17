# shell2.py
"""Volume 3: Unix Shell 2.
<Name> David Camacho
<Class> 
<Date> 10/20/22
"""
from glob import glob
import os
import subprocess


# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.

    Returns:
        matched_files (list): list of the filenames that matched the file
               pattern AND the target string.
    """
    
    FileList = glob("**/*" + file_pattern, recursive = True)

    MatchedFiles = []

    for file in FileList:
        with open(file, 'r') as myfile:
            contents = myfile.read()

        if target_string in contents:

            MatchedFiles.append(file)

    return MatchedFiles




# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """

    options = glob("**/*.*", recursive = True)

    file_sizes = []
    for file in options:
        size = os.path.getsize(file)
        file_sizes.append((size, file))

    sorted_files = sorted(file_sizes, key = lambda x: x[0])[::-1]


    return_list = []
    for i in range(n):
        return_list.append(sorted_files[i][1])

    
    smallest_file = return_list[-1]

    num_lines = subprocess.check_output(["wc", "-l", smallest_file]).decode().strip().split(" ")

    with open("smallest.txt", 'w') as file:
        file.write(num_lines[0])

    
    return return_list

    



    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter
