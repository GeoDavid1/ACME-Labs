# sql1.py
"""Volume 1: SQL 1 (Introduction).
<Name>
<Class>
<Date>
"""

import sqlite3 as sql
import csv
from matplotlib import pyplot as plt
import numpy as np

def examp1():
    conn = sql.connect("my_database.db")
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM MyTable")
    except sql.Error:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()
# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    
    # Drop necessary tables and create tables
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS MajorInfo")
        cur.execute("DROP TABLE IF EXISTS CourseInfo")
        cur.execute("DROP TABLE IF EXISTS StudentInfo")
        cur.execute("DROP TABLE IF EXISTS StudentGrades")
        cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)")
        cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
        cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
        cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")
    conn.close()

    # Insert MajorInfo Data
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        major_rows = [(1, 'Math'), (2, 'Science'), (3, 'Writing'), (4, 'Art')]
        cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", major_rows)
    conn.close()

    # Insert CourseInfo Data
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        ci_rows = [(1, 'Calculus'), (2, 'English'), (3, 'Pottery'), (4, 'History')]
        cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", ci_rows)
    conn.close()

    # Read in Student Info Data
    with open("student_info.csv", 'r') as infile:
        si_rows = list(csv.reader(infile))

    # Insert Student Info Data
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", si_rows)
    conn.close()

    # Read in Student Grades Data
    with open("student_grades.csv", 'r') as infile:
        sg_rows = list(csv.reader(infile))

    # Insert Student Grades Data
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?)", sg_rows)
    conn.close()

    #  Modify StudentInfo table so that values of -1 are NULL
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE StudentInfo SET MajorID = NULL WHERE MajorID = -1")
    conn.close()



    


# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """

    # Read earthquake data
    with open(data_file, 'r') as infile:
        rows = list(csv.reader(infile))

    # Drop USEarthquakes table and create new USEarthquakes table
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS USEarthquakes")
        cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")
        cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?)", rows)
    conn.close()

    # Remove rows from USEarthquakes that have a value 0;
    # Replace 0 values in Day, Hour, Minute, and Second columns with NULL values
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM USEarthquakes WHERE Magnitude == 0")
        cur.execute("UPDATE USEarthquakes SET Day = NULL WHERE Day == 0")
        cur.execute("UPDATE USEarthquakes SET Hour = NULL WHERE Hour == 0")
        cur.execute("UPDATE USEarthquakes SET Minute = NULL WHERE Minute == 0")
        cur.execute("UPDATE USEarthquakes SET Second = NULL WHERE Second == 0")
    conn.close()

# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """

    # Open SQL Connection
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        
        # Query database for all tuples of the form (StudentName, CourseName) where that student has an "A" or "A+"
        cur.execute("SELECT SI.StudentName, CI.CourseName "
                    "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                    "WHERE SG.StudentID = SI.StudentID AND CI.CourseID = SG.CourseID AND (Grade = 'A' OR Grade = 'A+')")
        
        results = cur.fetchall()
        
    conn.close()

    # Return results
    return results



# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    # Query the USEarthquakes table for the magnitudes of the earthquakes during the 19th century
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year >= 1800 AND Year <= 1899")
        
        first_results = cur.fetchall()

    conn.close()

    # Query the USEarthquakes table for the magnitudes of the earthquakes during the 20th century
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year >= 1900 AND Year <= 1999")

        second_results = cur.fetchall()

    conn.close()

    # Query average magnitude of all earthquakes in the database
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes")

        avg = cur.fetchall()
    conn.close()

    # Make numpy array of results
    first_results = np.array(first_results)
    second_results = np.array(second_results)

    # Histogram of the magnitudes of the earthquakes in the 19th century
    ax1 = plt.subplot(121)
    ax1.hist(first_results, bins = 9, range = [3,9])
    ax1.set_title("Magnitudes of the earthquakes: 19th century")
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel("Number of Earthquakes")

    # Histogram of the magnitudes of the earthquakes in the 20th century
    ax2 = plt.subplot(122)
    ax2.hist(second_results, bins = 9, range = [3,9])
    ax2.set_title("Magnitudes of the earthquakes: 20th century")
    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel("Number of Earthquakes")

    plt.show()

    # Return average
    average = avg[0][0]
    return average
    



