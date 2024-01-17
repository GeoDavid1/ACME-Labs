import pandas as pd
from scipy.stats import kstest
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = '/Users/user/Downloads/DellaCorte/Data/weighted_averages.xlsx'
data = pd.read_excel(file_path)

# Initialize a list to store normally distributed columns
normally_distributed_columns = []

# Iterate through each column
for column_name in data.columns:
    if column_name != 'participant_id':
        column_data = data[column_name]
        _, p_value = kstest(column_data, 'norm')
        
        # Define a significance level (e.g., 0.05)
        alpha = 0.05
        
        # Check if the p-value is greater than the significance level
        if p_value > alpha:
            normally_distributed_columns.append(column_name)
        

# Print out the columns that ARE normally distributed
print("Normally distributed columns:")
for column_name in normally_distributed_columns:
    print(column_name)

