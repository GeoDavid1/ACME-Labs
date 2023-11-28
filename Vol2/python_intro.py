# python_intro.py
"""Python Essentials: Introduction to Python.
<Name> David Camacho
<Class> ACME
<Date> 9/1/2022
"""


# Problem 1 (write code below)

from audioop import reverse
from typing import List


if __name__ == "__main__":
    print("Hello,world!")

# Problem 2


def sphere_volume(r):
    """ Return the volume of the sphere of radius 'r'.
        Use 3.14159 for pi in your computation.
        """


    return (4/3)*(3.14159)*(r**3)



print(sphere_volume(5))

#raise NotImplementedError("Problem 2 Incomplete")


# Problem 3

def isolate(a, b, c, d, e):



    print(a, end ='     ')
    print(b, end ='     ')
    print(c, end = ' ')
    print(d, end = ' ')
    print(e)

#print(isolate(1,2,3,4,5))

""" Print the arguments separated by spaces, but print 5 spaces on either
side of b.
"""
#raise NotImplementedError("Problem 3 Incomplete")


# Problem 4



def first_half(my_string):

    length_of_string = len(my_string) #Find length of string

    return my_string[0: ((length_of_string)//2)] #Return only the first half
    

#print(first_half("python"))

""" Return the first half of the string 'my_string'. Exclude the
middle character if there are an odd number of characters.

Examples:
    >>> first_half("python")
    'pyt'
    >>> first_half("ipython")
    'ipy'
"""
#raise NotImplementedError("Problem 4 Incomplete")


def backward(my_string):
    """ Return the reverse of the string 'my_string'."""

    
    i = 0   #Initialize first counter
    j = 0   #Initialize second counter
    reverse_string = ""  #Initialize the reverse string
    List_backward_string = [] #Initialize the list that will contain the characters of the reverse string

    len_my_string = len(my_string)




    while i < (len_my_string):  #Create the list of characters in reverse order
        List_backward_string.append(my_string[-1*i -1])
        i += 1

    

    
    while j < (len_my_string): #Concatenate all the characters in reverse order
        reverse_string = reverse_string + List_backward_string[j]
        j += 1

    
    return reverse_string

#print(backward("minnesota"))
    
    








#raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def list_ops():

    Problem_5_List = ["bear", "ant", "cat", "dog"]

    Problem_5_List.append("eagle")      #Append "eagle"
    #print(Problem_5_List) 

    Problem_5_List[2] = "fox"           #Replace the entry at index 2 with "fox"
    #print(Problem_5_List) 

    Problem_5_List.pop(1)          # Remove (or pop) the entry at index 1
    #print(Problem_5_List)

    Problem_5_List.sort(reverse= True)    # Sort the list in reverse alphabetical order
    #print(Problem_5_List)

    index_eagle = Problem_5_List.index("eagle") #Find index of eagle

    Problem_5_List[index_eagle] = "hawk"   #Replace "eagle" with "hawk"
    #print(Problem_5_List)

    Problem_5_List[-1] = Problem_5_List[-1] + "hunter"  #Add the string "hunter" to the last entry in the list

    return Problem_5_List

#print(list_ops())


""" Define a list with the entries "bear", "ant", "cat", and "dog".
Perform the following operations on the list:
    - Append "eagle".
    - Replace the entry at index 2 with "fox".
    - Remove (or pop) the entry at index 1.
    - Sort the list in reverse alphabetical order.
    - Replace "eagle" with "hawk".
    - Add the string "hunter" to the last entry in the list.
Return the resulting list.

Examples:
    >>> list_ops()
    ['fox', 'hawk', 'dog', 'bearhunter']
"""
#raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def pig_latin(word):

    if 'a' in word[0] or 'e' in word[0] or 'i' in word[0] or 'o' in word[0] or 'u' in word[0]: #Check to see if first character is a vowel or not

        word = word + "hay"  #If a vowel, add "hay"
    
    else:
        temp_char = word[0]   #Set up a temporary variable to store word[0]
        word = word[1:] + temp_char + "ay"  #Take the first character of word, move it to the end and add "ay"

    return word

#print(pig_latin("zebra"))


""" Translate the string 'word' into Pig Latin, and return the new word.

Examples:
    >>> pig_latin("apple")
    'applehay'
    >>> pig_latin("banana")
    'ananabay'
"""
#raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def palindrome():

    max_palindrome = 0    #Set initial maximum palindromic number
    Products_List = []    #Set list of all products of two 3-digit numbers
    i = 100              #First counting index
    j = 100              #Second counting index

    
    while i < 1000:   #Nested while loop runs through all two 3-digit number combinations

        j = 1           # Reset the second counting index

        while j < 1000:
            Products_List.append(i*j) #Append all possible products to Products_List
            j +=1
            
        i += 1


    for entry in Products_List: #Go through all entries in Products_List
        str_entry = str(entry)  # Make the entry a string
        reversed_str_entry = str(entry)[::-1] #Make the reverse of the entry a string
        if str_entry == reversed_str_entry: #Check to see if the entry is a palindrome
            if entry > max_palindrome:      # Check to see if this palindrome is greater than the greatest we have found so far
                max_palindrome = entry     # If greater, max_palindrome is replaced by entry
    
    return max_palindrome

    





""" Find and retun the largest panindromic number made from the product
of two 3-digit numbers.
"""
#raise NotImplementedError("Problem 7 Incomplete")

# Problem 8
def alt_harmonic(n):

    i = 1           #Initialize counter
    Alt_Sum = 0     #Initialize the Summation
    List_kth_terms = []  #Initialize list of terms in summation


    def kth_value(k):    #Define a function that spits out the kth term in the summation
        temp_kth_value = (1/k)*(-1)**(k+1)
        return temp_kth_value

    while i < (n+1):   #Append all kth terms to the List_kth_terms
        List_kth_terms.append(kth_value(i))
        i += 1
    
    Alt_Sum = sum(List_kth_terms)   #Sum the kth terms up

    return Alt_Sum



""" Return the partial sum of the first n terms of the alternating
harmonic series, which approximates ln(2).
"""
#raise NotImplementedError("Problem 8 Incomplete")
