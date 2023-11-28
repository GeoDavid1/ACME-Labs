# standard_library.py
"""Python Essentials: The Standard Library.
<Name> David Camacho
<Class> ACME
<Date>9/6/22
"""

import sys
import time
import box

# Problem 1
def prob1(L):

    minimum = min(L)  #Find minimum
    maximum = max(L)  #Find maximum
    avg = sum(L)/len(L)  #Find average

    return minimum, maximum, avg   #Return desired value

    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
 
   """


    #raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():

    n1 = 1                  #Check to see if integers are mutable
    n2 = n1                 #...
    n1 += 3                 #...

    if n1 == n2:                          #...
        print("Integers are mutable")     #...

    else:                                   #...
        print("Integers are immutable")     # Integers are immutable

    str1 = '1'                              #Check to see if strings are mutable
    str2 = str1                             #...
    str1 = '2'                              #...

    if str1 == str2:                        #...
        print("Strings are mutable")        #...
    
    else:                                   #...
        print("Strings are immutable")      #Strings are immutable

    
    list_1 = ["apple", "banana", "orange"]  #Check to see if lists are mutable
    list_2 = list_1                         #...

    list_1[1] = ["cherry"]                  #...

    if(list_1 == list_2):                   #...
        print("Lists are mutable")          #Lists are mutable

    else:
        print("Lists are immutable")        #...


    tuple_1 = ("apple", "banana", "cherry")   #Check to see if tuples are mutable
    tuple_2 = tuple_1                         #...
    tuple_1 += (1,)                           #...

    if(tuple_1 == tuple_2):                   #...
        print("Tuples are mutable")           #...

    else:
        print("Tuples are immutable")         #Tuples are immutable

    set_1 = {"apple", "banana", "cherry"}
    set_2 = set_1
    set_1 = {"apple", "banana", "cherry", "dragonfruit"}

    if(set_1 == set_2):
        print("Sets are immutable")

    else:
        print("Sets are mutable")




    

    # This is my answer

print(prob2())



    #raise NotImplementedError("Problem 2 Incomplete")

#print(prob2())



# Problem 3

import calculator as calc

def hypot(a, b):

    a_squared = calc.product(a,a)  #Find a_squared
    b_squared = calc.product(b,b)  #Find b_squred
    hypotenuse_squared = calc.sum(a_squared, b_squared) #Find hypotenuse_squared
    hypotenuse = calc.sqrt(hypotenuse_squared) #Find hypotenuse

    return hypotenuse
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    raise NotImplementedError("Problem 3 Incomplete")

#print(hypot(5,12))

# Problem 4

import itertools as iter
def power_set(A):

    Power_Set = [set()]  #Initialize the Power Set as nothing
    len_A = len(A)       #Find length of A
    i = 1                #Initialize counter
    while i <= len_A:  # While loop creates the power set
        j = 0           #Reset counter
        tempList = list(iter.combinations(A,i)) # Make all possible combinations of length i
        while j < len(tempList):   # While loop appends all these combinations to the Power Set

            Power_Set.append(tempList[j])
            j+=1                         #Move up the counter
        
        i +=1                             #Move up the counter

    for i in range(len(Power_Set)):
        Power_Set[i] = set(Power_Set[i])

    return Power_Set


print(power_set("string"))


   

#print(power_set(['a', 'b', 'c', 'r']))


# Problem 5: Implement shut the box.
import random

def shut_the_box(player, timelimit):


    Initial_Time = time.time()                 #Initialize Initial Time
    Numbers_Left = [1,2,3,4,5,6,7,8,9]         #Initialize What Numbers Are In The Game
    Game_Over = False                          #Initialize boolean Game_Over



    while (len(Numbers_Left) != 0):           # Run this loop while there are still numbers left over

        if sum(Numbers_Left) > 6:             

            temp_Dice_Roll = random.randint(1,6) + random.randint(1,6)  #Sum is greater than 6, thus roll two dice
            Temp_Time = time.time()                                     #Measure Time
        else:

            temp_Dice_Roll = random.randint(1,6)                        #Sum is 6 or less, thus roll one die
            Temp_Time = time.time()                                     #Measure Time

        

        Time_Left = timelimit - (Temp_Time - Initial_Time)              # Compute how much time is left

        print("Numbers left: " + str(Numbers_Left))                     # Print out Numbers Left
        print("Roll: " + str(temp_Dice_Roll))                           # Print out Roll
        if(box.isvalid(temp_Dice_Roll, Numbers_Left) == False):         # See if you can compute the Roll by summing the Numbers Left (If so, Game Over)
            Game_Over = True
            break

        print("Seconds left: "  + str(round(Time_Left, 2)))             # Print out Seconds Left
        Player_Input = input("Numbers to eliminate: ")                  # User input for Numbers to eliminate

        Temp_Time = time.time()                                         # Measure time again
        Time_Left = timelimit - (Temp_Time - Initial_Time)              # Compute how much time is left
        if(Time_Left <= 0):                                             # See if Time has run out (If so, Game Over)
            Game_Over = True
            break


        

        if(box.parse_input(Player_Input, Numbers_Left) != []):         #Check to see that the numbers in player's input are in the Numbers Left
            tempSum = 0                                                #If so, sum the player's input numbers
            List_Played = box.parse_input(Player_Input, Numbers_Left)  
            for i in List_Played:
                tempSum += i


            while(tempSum != temp_Dice_Roll):                          # Run a while loop to force the player to put in numbers that sum up to the value of the roll
                print("Invalid input")                                 # Print this collection of statements
                print(" ")
                Temp_Time = time.time()
                Time_Left = timelimit - (Temp_Time - Initial_Time)
                print("Seconds left: " + str(round(Time_Left, 2)))
                Player_Input = input("Numbers to eliminate: ")
                
                tempSum = 0                                                # Reset tempSum
                List_Played = box.parse_input(Player_Input, Numbers_Left)  # Recalculate tempSum and hope that it is equal to the Dice Roll
                for i in List_Played:
                    tempSum += i

            
            print(" ")                                                # Remove numbers played from the Numbers_Left list                        
            
        
        else:                                                         # Run this else statement if the player's input numbers are NOT in the Numbers Left list
            while (box.parse_input(Player_Input, Numbers_Left) == []):  # Force the player to put numbers that are in the Numbers Left list
                print("Invalid input")                                  # Print this collection of statements
                print(" ")
                Temp_Time = time.time()
                Time_Left = timelimit - (Temp_Time - Initial_Time)
                print("Seconds left: " + str(round(Time_Left, 2)))
                Player_Input = input("Numbers to eliminate: ")         #  Replace the Player Input with the new numbers the player typed in

                tempSum = 0                                                # Reset tempSum
                List_Played = box.parse_input(Player_Input, Numbers_Left)  # Recalculate tempSum and hope that it is equal to the Dice Roll
                for i in List_Played:
                    tempSum += i
                                                    

                
        for i in List_Played:
            Numbers_Left.remove(i)       

        
    if (Game_Over == True):    # Run this end-of-game statement if the player lost
        print("Game over!")
        print(" ")
        End_Sum = 0
        for k in Numbers_Left:      
            End_Sum += k

        print("Score for player " + str(player) + ": " + str(End_Sum) + " points" )             #Player's Score
        Temp_Time = time.time() 
        Time_Played = round((Temp_Time - Initial_Time), 2)
        print("Time played: " + str(Time_Played) + " seconds")                                  #Amount of time played
        print("Better luck next time >:)")

    elif (Game_Over == False):              # Run this end-of-game statement if the player won
        print(" ")
        print("Score for player " + str(player) + ": " + str(0) + "points" )            #Player's Score (which is 0)
        Temp_Time = time.time() 
        Time_Played = round((Temp_Time - Initial_Time), 2)
        print("Time played: " + str(Time_Played) + " seconds")                          #Amount of time played
        print("Congratulations!! You shut the box!")






            





#print(shut_the_box('David',60))
         

if __name__ == "__main__":

    if len(sys.argv) == 3:

         player = sys.argv[1]
         timelimit = int(sys.argv[2])
         shut_the_box(player, timelimit)  

    else:

        print("Unable to start the game")

    
    #raise NotImplementedError("Problem 5 Incomplete")
