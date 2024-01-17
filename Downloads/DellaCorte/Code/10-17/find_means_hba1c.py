import pandas as pd

# Read the 'Final_Data.xlsx' file into a DataFrame
final_data = pd.read_excel('/Users/user/Downloads/DellaCorte/Final_Data.xlsx')

# Group the data by 'ecid_hba1c_tertile' and calculate the means
means_by_tertile = final_data.groupby('ecid_hba1c_tertile').mean()

# Define the output Excel file path
output_file = '/Users/user/Downloads/DellaCorte/Tertiles.xlsx'

# Define the sheet name
sheet_name = 'Hba1c_Tertile_Raw_Data'

# Save the means to the Excel file
means_by_tertile.to_excel(output_file, sheet_name=sheet_name)






