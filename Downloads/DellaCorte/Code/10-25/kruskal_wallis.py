import pandas as pd
import numpy as np

# Load data from 'weighted_averages.xlsx'
file_path = '/Users/user/Downloads/DellaCorte/Data/weighted_averages.xlsx'
data = pd.read_excel(file_path)

# Specify the column for which you want to calculate tertiles
tertile_column_nmes = 'Weighted Average for intake_nmes'
tertile_column_hba1c = 'Weighted Average for ecid_hba1c_percent'

# Calculate tertiles for the specified columns manually
nmes_values = data[tertile_column_nmes]
hba1c_values = data[tertile_column_hba1c]

nmes_tertiles = np.percentile(nmes_values, [0, 33.33, 66.67, 100])
hba1c_tertiles = np.percentile(hba1c_values, [0, 33.33, 66.67, 100])

# Assign tertiles to the DataFrame
data['Tertile intake_nmes'] = pd.cut(nmes_values, bins=nmes_tertiles, labels=['1', '2', '3'])
data['Tertile ecid_hba1c_percent'] = pd.cut(hba1c_values, bins=hba1c_tertiles, labels=['1', '2', '3'])

# Print the updated DataFrame with the new tertile columns
print(data[['Tertile intake_nmes', 'Tertile ecid_hba1c_percent']])

# Export as xlsx this data
# Export the updated DataFrame to a new Excel file
output_file_path = '/Users/user/Downloads/DellaCorte/Data/weighted_averages.xlsx'
data.to_excel(output_file_path, index=False)

