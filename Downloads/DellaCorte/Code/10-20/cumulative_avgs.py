import pandas as pd
from tqdm import tqdm

# Define the path to your Excel file
PATH = '/Users/user/Downloads/DellaCorte/XSLX/Merged.xlsx'

# Specify the sheet name within the Excel file
sheet_name = 'Merged_All_Data'

# Load the data from the Excel file into a Pandas DataFrame
df = pd.read_excel(PATH, sheet_name)

intake_variables = ['participant_id', 'CID', 'intake_protein', 'intake_fat', 'intake_carbohydrate',
                    'intake_starch', 'intake_energy_kilocalories',
                    'intake_sucrose', 'intake_fibre_englyst',
                    'intake_fibre_southgate', 'intake_total_sugars',
                    'intake_nmes', 'intake_intrinsic_sugars',
                    'intake_satd_fa', 'intake_cholesterol',
                    'intake_fructose', 'intake_glucose']

anthro_data = ['participant_id', 'CID', 'age_at_study', 'ecid_weight_recorded',
               'ecid_height_recorded', 
               'cid_moissl_fat_mass', 'cid_moissl_fat_mass_percent',
               'ecid_sbp', 'ecid_dbp', 'ecid_bpm']

blood_glucose_data = ['participant_id', 'CID', 'ecid_hba1c_percent', 'ecid_ldl',
                      'ecid_hdl', 'ecid_trig']

# Create DataFrames for each category of columns
intake_df = df[intake_variables]
anthro_df = df[anthro_data]
blood_glucose_df = df[blood_glucose_data]

# Define the common weights for cumulative averages
weights = [1/8, 1/8, 1/4, 1/2]

# Group by participant ID
grouped = intake_df.groupby('participant_id')

print(grouped)

# Initialize an empty DataFrame to store the cumulative averages
cumulative_avg_df = pd.DataFrame(columns=['participant_id'])

# Create a tqdm instance to track progress
progress_bar = tqdm(total=len(intake_variables) - 2)  # Exclude 'participant_id' and 'CID'

# Calculate the cumulative average for each intake variable
for variable in intake_variables[2:]:
    cumulative_avg_df[f'cumulative_avg_{variable}'] = grouped.apply(lambda x: sum(x[x['CID'] == i][variable] * weight for i, weight in enumerate(weights)))

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Reset the index and drop it
cumulative_avg_df.reset_index(inplace=True, drop=True)

# Print the cumulative average DataFrame
print(cumulative_avg_df)

