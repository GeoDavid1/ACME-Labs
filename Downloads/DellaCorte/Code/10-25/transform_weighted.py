import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from 'weighted_averages.xlsx' and specify the sheet
file_path = '/Users/user/Downloads/DellaCorte/Data/weighted_averages.xlsx'
sheet_name = 'Mixed Weighted Averages'
data = pd.read_excel(file_path, sheet_name)

# Create a copy of the DataFrame to store the square root values
sqrt_data = data.copy()

# Iterate through each column and calculate the square root
for column_name in data.columns:
    sqrt_data[column_name] = np.sqrt(data[column_name])


# Rename the columns: Sqrt of [column_name]
sqrt_data.columns = ['Sqrt of ' + col if col != 'participant_id' else col for col in sqrt_data.columns]

# Define the output file path for the new Excel file
output_file_path = 'sqrt_mixed_averages.xlsx'  # Updated output file path

# Write the DataFrame to the Excel file
sqrt_data.to_excel(output_file_path, sheet_name='Sqrt Mixed Averages', index=False)

