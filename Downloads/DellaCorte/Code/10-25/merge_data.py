import pandas as pd

# Load data from the individual Excel files
blood_data = '/Users/user/Downloads/DellaCorte/Data/blood_weighted_averages.xlsx'
anthro_data = '/Users/user/Downloads/DellaCorte/Data/anthro_weighted_averages.xlsx'
intake_data = '/Users/user/Downloads/DellaCorte/Data/intake_weighted_averages.xlsx'

try:
    df_blood = pd.read_excel(blood_data)
    df_anthro = pd.read_excel(anthro_data)
    df_intake = pd.read_excel(intake_data)
except Exception as e:
    print("An error occurred while loading data:", e)

# Merge the dataframes using 'participant_id' as the key
merged_df = df_blood.merge(df_anthro, on='participant_id', how='inner')
merged_df = merged_df.merge(df_intake, on='participant_id', how='inner')

# Save the merged dataframe to a new Excel sheet
output_path = '/Users/user/Downloads/DellaCorte/Data/weighted_averages.xlsx'
merged_df.to_excel(output_path, index=False)

print("Merged data saved to weighted_averages.xlsx")
