import pandas as pd

# Load data from 'Merged_Data_1.xlsx' and the specified sheet
merged_data_1 = pd.read_excel('/Users/user/Downloads/DellaCorte/Data/weighted_averages.xlsx')

print(merged_data_1.columns)
# Find tertiles of the 'Weighted Average for intake_nmes' column and print them
tertiles_intake_nmes = merged_data_1['Weighted Average for intake_nmes'].quantile([0, 1/3, 2/3, 1])

print("Tertiles for 'Weighted Average for intake_nmes':")
print(tertiles_intake_nmes)

# Add a new column classifying each row as 1st, 2nd, or 3rd tertile for 'Weighted Average for intake_nmes'
merged_data_1['intake_nmes_tertile'] = pd.cut(merged_data_1['Weighted Average for intake_nmes'], bins=tertiles_intake_nmes, labels=[1, 2, 3])

# List of columns to calculate averages for (excluding 'participant_id' and 'intake_nmes_tertile')
columns_to_average = [col for col in merged_data_1.columns if col not in ['participant_id', 'intake_nmes_tertile']]

# Find the averages also of the intake of the nmes by tertile, as well
# Dictionary to store the results
averages_by_tertile = {}

# Loop through columns and calculate averages for each tertile
for col in columns_to_average:
    averages = merged_data_1.groupby('intake_nmes_tertile')[col].mean()
    averages_by_tertile[col] = averages

# Convert the dictionary to a DataFrame
averages_df = pd.DataFrame(averages_by_tertile)

# Save the averages to a new Excel file
output_file = '/Users/user/Downloads/DellaCorte/Cum_Final_Data.xlsx'
sheet = 'Averages_by_NMES_Tertile'
averages_df.to_excel(output_file, sheet_name=sheet)
