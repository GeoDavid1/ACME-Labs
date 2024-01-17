import pandas as pd

# Replace the file path with the actual path to your Excel file
file_path = '/Users/user/Downloads/DellaCorte/sqrt_mixed_averages.xlsx'

# Read data from the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)

# Now you can work with the DataFrame 'df' to analyze and manipulate the data.


X_variable = 'Sqrt of age_at_study' # column to X data
y_variable = 'Sqrt of ecid_hba1c_percent'

def get_linear_regression