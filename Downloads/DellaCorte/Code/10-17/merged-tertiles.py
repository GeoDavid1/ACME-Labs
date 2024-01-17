import pandas as pd

# Load data from 'Merged_Data_1.xlsx' and the specified sheet
merged_data_1 = pd.read_excel('/Users/user/Downloads/DellaCorte/XSLX/Merged.xlsx', sheet_name='Means_By_CID')

# Assuming you want to average data for each 'participant_id'
# Group the data by 'participant_id' and calculate the mean of other columns while excluding NaN values
averaged_data = merged_data_1.groupby('participant_id').mean().reset_index()

# Print the result
print(averaged_data)

# Save the result in an Excel file called 'Final Data'
averaged_data.to_excel('Final_Data.xlsx', index=False)
