import pandas as pd

# Define the path to your Excel file
PATH = '/Users/user/Downloads/DellaCorte/XSLX/Merged.xlsx'
sheet_name = 'Merged_All_Data'

# Load the data from the Excel file into a Pandas DataFrame
df = pd.read_excel(PATH, sheet_name)

# Specify the columns of interest - all columns that have numerical data
numerical_columns = df.select_dtypes(include=['number']).columns

# Calculate the average for the specified columns by 'participant_id'
averages = df.groupby('participant_id')[numerical_columns].mean()

# Print the resulting DataFrame with the averages
print(averages)

# Save the result as an Excel (.xlsx) file
result_file_path = 'averages_by_participant.xlsx'
averages.to_excel(result_file_path)
print(f'Averages saved to {result_file_path}')



# Get all rows that have '1' in these columns: ['ecid_breastfeeding', 'ecid_diagnosed_condition']
columns_to_check = ['ecid_breastfeeding', 'ecid_diagnosed_condition', 'ecid_diabetes_1']
rows_with_ones = df[df[columns_to_check].eq(1).any(axis=1)]

# Print the resulting DataFrame with rows that have '1' in the specified columns
print(rows_with_ones)
