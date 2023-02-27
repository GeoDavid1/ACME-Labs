# regular_expressions.py
"""Volume 3: Regular Expressions.
<Name>
<Class>
<Date>
"""

import re
import datetime

def examp1():
    pattern = re.compile("cat")
    print(bool(pattern.search("cat")))
    print(bool(pattern.match("catfish")))
    print(bool(pattern.match("fishcat")))
    print(bool(pattern.search("hat")))

    print(bool(re.compile("cat"). search("catfish")))
    print(bool(re.search("cat", "catfish")))


# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    
    # Compile and return regular expression pattern object
    pattern = re.compile("python")
    return pattern

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    
    # Compile and return regular expression pattern object
    return re.compile(r"\^\{@}\(\?\)\[%]\{\.}\(\*\)\[_]\{&}\$")

def examp3():

    fish = re.compile(r"^(one|two) fish$")
    for test in ["one fish", "two fish", "red fish", "one two fish"]:
        print(test + ':', bool(fish.search(test)))

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """

    # Compile needed regular expression pattern object and return it
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")


# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """

    # Compile needed regular expression pattern object and return it
    return re.compile(r"^[a-zA-Z_]\w*\s*(=\s*(\d*(\.\d*)?|'[^']*'|[a-zA-Z_]\w*))?$")
    
def examp4():

    p1, p2 = re.compile(r"^[a-z][^0-7]$"), re.compile(r"^[abcA-C][0-27-9]$")
    for test in ["d8", "aa", "E9", "EE", "d88"]:
        print(test + ':', bool(p1.search(test)), bool(p2.search(test)))

def examp4part2():
    pattern = re.compile(r"^.\d.$")
    for test in ["a0b", "888", "n2%", "abc", "cat"]:
        print(test + ':', bool(pattern.search(test)))

    pattern2 = re.compile(r"^[a-zA-Z][a-zA0Z]\d..$")
    for test in ["tk421", "bb8!?", "JB007", "Boba?"]:
        print(test + ':', bool(pattern2.search(test)))

def part3():
    pattern3 = re.compile(r"^a{3}$")
    for test in ["aa", "aaa", "aaaa", "aba"]:
        print(test + ':', bool(pattern3.search(test)))

def part4():
    pattern4 = re.compile(r"a{3}")
    for test in ["aaa", "aaaa", "aaaaa", "aaaab"]:
        print(test + ':', bool(pattern4.search(test)))



    

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """

    #Compile needed regular expression pattern object and return it
    return re.compile(r"((if|elif|else|for|while|try|except|finally|with|def|class).*)").sub(r"\1:", code)



def examp5():

    pig_latin = re.compile(r"\bcat(\w*)")
    target = "Let's catch some catfish for the cat"

    print(pig_latin.sub(r"at\1clay", target)) # \1 = (\w*) from the expression

def examp52():

    target = "<abc><def><ghi>"    

    # Match angle brackets and anything in between.
    
    greedy = re.compile(r"^.*>$")
    print(greedy.findall(target))

    nongreedy = re.compile(r"<.*?>") # Non-greedy
    print(nongreedy.findall(target))

    pattern1 = re.compile("^W")
    pattern2 = re.compile("^W", re.MULTILINE)
    print(bool(pattern1.search("Hello\nWorld")))
    print(bool(pattern2.search("Hello\nWorld")))




# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """

    # Initialize Dictionary; Read in Contents
    needed_dict = {}
    with open(filename, 'r') as myfile:
        contents = myfile.readlines()


    # Create patterns to find name, birthday, phone, and email
    pattern_name = re.compile(r"^[A-Za-z]+(\s[A-Za-z]\.)?(\s[A-Za-z]+)")
    pattern_birthday = re.compile(r"([\d]*)\/([\d]*)\/([\d]*)")
    pattern_phone = re.compile(r"([\d-]?)(\()?([\d]{3})(\))?(-)?([\d]{3})(-[\d]{4})")
    pattern_email = re.compile(r"(\b)(\S)*(@)(\S)*(\b)")

    # Find the name, date, phone, and email in each line
    for line in contents:
        new_dict = {}
        name = pattern_name.search(line).group()
        date = pattern_birthday.findall(line)
        phone = pattern_phone.findall(line)
        email = pattern_email.search(line)

        # Format birthdate
        if date:
            date = date[0]
            month = date[0] if len(date[0]) == 2 else '0' + date[0]
            day = date[1] if len(date[1]) == 2 else '0' + date[1]
            year = date[2] if len(date[2]) == 4 else '20' + date[2]
            newbday = month + "/" + day + "/" + year
            new_dict["birthday"] = newbday

        # Set birthdate to None if there is none
        else:
            new_dict["birthday"] = None

        # Format email
        if email:
            new_dict["email"] = email.group()

        # Set email to None if there is none
        else:
            new_dict["email"] = None

        # Format phone number
        if phone:
            phone = [phone[0][a] for a in [2,5,6]]
            new_phone = "(" + phone[0] + ")" + phone[1] + phone[2]
            new_dict["phone"] = new_phone

        # Set phone number to None if there is none
        else:
            new_dict["phone"] = None

        # Make Dictionary inside of dictionary
        needed_dict[name] = new_dict

    # Return dictionary
    return needed_dict


            
    








def examp6():

    pattern = re.compile("(\w*) (?:fish|dish)")
    print(pattern.findall("red dish, blue dish, one fish, two fish"))


    pattern2 = re.compile("(\w*) (fish|dish)")
    print(pattern2.findall("dish red, blue dish, one fish, two fish"))

    pattern3 = re.compile("(\w*) (?:fish|dish)")
    print(pattern3.findall("red dish, blue fish, one fish, two fish"))

