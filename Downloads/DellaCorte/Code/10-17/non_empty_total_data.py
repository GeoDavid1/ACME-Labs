import pandas as pd

# Replace the file path with your actual file path
file_path = "/Users/user/Downloads/DellaCorte/Data/Dietary_Data_with_Residuals-10-11.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Remove columns that start with 'Sqrt'
df = df.loc[:, ~df.columns.str.startswith('Sqrt')]


# Calculate the proportion that are NOT NaN values per column
nan_proportion = (1 - df.isna()).mean()


# Print the result to a text file
output_file = 'Total Data - Nonempty.txt'
nan_proportion.to_csv(output_file, header=True, sep='\t')

# Optionally, you can also print the result to the console
print(nan_proportion)




