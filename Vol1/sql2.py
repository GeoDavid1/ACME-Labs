# solutions.py
"""Volume 1: SQL 2.
<Name> David Camacho
<Class> ACME
<Date>
"""
import sqlite3 as sql

def examp1():
    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT * "
                    "FROM StudentInfo AS SI INNER JOIN MajorInfo AS MI "
                    "ON SI.MajorID == MI.MajorID;")
        

        results = cur.fetchall()

    conn.close()

    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT * "
                    "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                    "ON SI.MajorID == MI.MajorID")
        
        results2 = cur.fetchall()

    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT CI.CourseName, SG.Grade "
                    "FROM StudentInfo AS SI "
                    "INNER JOIN CourseInfo AS CI, StudentGrades SG "
                    "ON SI.StudentID == SG.StudentID AND CI.CourseID == SG.CourseID "
                    "WHERE SI.StudentName == 'Kristopher Tran' ")
        
        results3 = cur.fetchall()



    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT StudentID, COUNT(*) "
                    "FROM StudentGrades "
                    "GROUP BY StudentID ")
        
        results4 = cur.fetchall()

    conn.close()

    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT SI.StudentName, COUNT(*) "
                    "FROM StudentGrades AS SG INNER JOIN StudentInfo AS SI "
                    "ON SG.StudentID == SI.StudentID "
                    "GROUP BY SG.StudentID")
        
        results5 = cur.fetchall()
    
    conn.close()

   

    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT SI.StudentName, COUNT(*) as num_courses "
                    "FROM StudentGrades AS SG INNER JOIN StudentInfo AS SI "
                    "ON SG.StudentID == SI.StudentID "
                    "GROUP BY SG.StudentID "
                    "HAVING num_courses == 3")
        
        results6 = cur.fetchall()

    conn.close()


    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT SI.StudentName, COUNT(*) AS num_courses "
                    "FROM StudentGrades AS SG INNER JOIN StudentInfo AS SI "
                    "ON SG.StudentID == SI.StudentID "
                    "GROUP BY SG.StudentID "
                    "ORDER BY num_courses DESC, SI.StudentName ASC ")
        
        results7 = cur.fetchall()

    conn.close()

    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT StudentName FROM StudentInfo "
                    "WHERE StudentName LIKE '%i%' ")
        
        results8 = cur.fetchall()

    print(results8)

    conn.close()

    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT StudentName, CASE MajorID "
                    "WHEN 1 THEN 'Mathematics' "
                    "WHEN 2 THEN 'Soft Science' "
                    "WHEN 3 THEN 'Writing and Editing' "
                    "WHEN 4 THEN 'Fine Arts' "
                    "ELSE 'Undeclared' END "
                    "FROM StudentInfo "
                    "ORDER BY StudentName ASC ")
        
        
        results9 = cur.fetchall()
    
    conn.close()


    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT StudentName, CASE "
                    "WHEN MajorID IS NULL THEN 'Undeclared' "
                    "ELSE 'Declared' END "
                    "FROM StudentInfo "
                    "ORDER BY StudentName ASC;")
        
        results10 = cur.fetchall()

    conn.close()
        

    print(results10)

    with sql.connect("students.db") as conn:
        
        cur = conn.cursor()
        
        cur.execute("SELECT majorstatus, COUNT(*) AS majorcount "
                     "FROM ( "
                     "SELECT StudentName, CASE "
                     "WHEN MajorID IS NULL THEN 'Undeclared' "
                     "ELSE 'Declared' END AS majorstatus "
                     "FROM StudentInfo) "
                    "GROUP BY majorstatus "
                    "ORDER BY majorcount DESC ")
        
        results11 = cur.fetchall()

    print(results11)

    with sql.connect("students.db") as conn:

        cur = conn.cursor()

        cur.execute("SELECT SI.StudentName, AVG(SG.gradeisa) "
                    "FROM ("
                        "SELECT StudentID, CASE Grade "
                            "WHEN 'A+' THEN 1 "
                            "WHEN 'A' THEN 1 "
                            "WHEN 'A-' THEN 1 "
                            "ELSE 0 END AS gradeisa "
                        "FROM StudentGrades) AS SG "
                    "INNER JOIN StudentInfo AS SI "
                    "ON SG.StudentID == SI.StudentID "
                    "GROUP BY SG.StudentID ")
        
        results12 = cur.fetchall()
        print()
        print(results12)
        

        






# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    
    # Accept the name of a database
    with sql.connect(db_file) as conn:
        cur = conn.cursor()

        # Query the database for the list of the names of students who have a B grade in any course
        cur.execute("SELECT SI.StudentName "
                    "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG "
                    "ON SI.StudentID == SG.StudentID "
                    "WHERE SG.Grade == 'B'")
        
        # Get results
        results = cur.fetchall()

    # Stop SQL connection
    conn.close()

    # Return a list of strings
    string_list = [''.join(item) for item in results]
    return string_list


# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    
    # Accept the name of a database
    with sql.connect(db_file) as conn:
        cur = conn.cursor()

        # Query the database for all tuples of the form (Name, MajorName, Grade)
        cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                    "FROM StudentInfo AS SI  LEFT OUTER JOIN MajorInfo AS MI "
                    "ON SI.MajorID == MI.MajorID "
                    "INNER JOIN StudentGrades AS SG "
                    "ON SI.StudentID == SG.StudentID "
                    "WHERE SG.CourseID == '1' ")
        
        # Get results
        results = cur.fetchall()

    # Stop SQL connection
    conn.close()

    # Return results
    return results




# Problem 3
def prob3(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    
    # Accept the name of a dabatabse
    with sql.connect(db_file) as conn:
        cur = conn.cursor()

        # Query the given database for tuples of the form (MajorName, N) where N is the number of students in the specified major
        cur.execute("SELECT MI.MajorName, COUNT(*) AS num_students "
                    "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                    "ON SI.MajorID == MI.MajorID "
                    "GROUP BY MI.MajorName "
                    "ORDER BY num_students DESC, MI.MajorName ASC ")
        
        # Get results
        results = cur.fetchall()

    # Close SQL connection
    conn.close()

    # Return results
    return results
        


# Problem 4
def prob4(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    
    # Accept the name of a database file
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        
        # Query the database for tuples of the form (StudentName, N, GPA) where N is the number of courses that the specified student is enrolled in
        cur.execute("""SELECT SI.StudentName, COUNT(*), AVG(CASE SG.Grade
                            WHEN 'A+' THEN 4.0 
                            WHEN 'A' THEN 4.0 
                            WHEN 'A-' THEN 3.7
                            WHEN 'B+' THEN 3.4
                            WHEN 'B' THEN 3.0
                            WHEN 'B-' THEN 2.7
                            WHEN 'C+' THEN 2.4
                            WHEN 'C' THEN 2.0
                            WHEN 'C-' THEN 1.7
                            WHEN 'D+' THEN 1.4
                            WHEN 'D' THEN 1.0
                            WHEN 'D-' THEN 0.7
                            ELSE 0 END) AS gpa
                        FROM StudentGrades AS SG INNER JOIN StudentInfo AS SI 
                        ON SG.StudentID == SI.StudentID
                        GROUP BY SG.StudentID ORDER BY gpa DESC""")

        # Get results
        results = cur.fetchall()
    
    # Return results
    return results


# Problem 5
def prob5(db_file="mystery_database.db"):
    """Use what you've learned about SQL to identify the outlier in the mystery
    database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): outlier's name, outlier's ID number, outlier's eye color, outlier's height
    """

    # Accept the name of the database
    with sql.connect(db_file) as conn:

        # Query the database for information
        cur = conn.cursor()
        cur.execute("SELECT * FROM table_1;")
        cur.execute("SELECT * FROM table_2;")
        cur.execute("SELECT * FROM table_3;")
        cur.execute("SELECT * FROM table_4;")

        # Find Name and ID from description
        cur.execute("SELECT * FROM table_2 WHERE description LIKE '%before finally accepting%';")
        results2 = cur.fetchall()
        Name = "William T. Riker"
        ID = results2[0][0]

        # Find Eye Color
        cur.execute("SELECT * FROM table_1 WHERE name LIKE '%William T. Riker%'")
        name_results = cur.fetchall()
        Eye_Color = "Hazel-blue"

        # Find Height
        cur.execute("SELECT * FROM table_3 WHERE gender LIKE '%Male%' AND eye_color LIKE '%Hazel-blue%'")
        heights = cur.fetchall()
        Height = '1.93'

        # Return Needed List 
        list_needed = [Name, ID, Eye_Color, Height]
        return list_needed


    




