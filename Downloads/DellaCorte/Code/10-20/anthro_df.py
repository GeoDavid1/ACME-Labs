import pandas as pd

import pandas as pd

# Define the path to your Excel file
PATH = '/Users/user/Downloads/DellaCorte/Merged.xlsx'

# Specify the sheet name within the Excel file
sheet_name = 'Means_By_CID'

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
               'ecid_height_recorded', 'cid_moissl_fat_free_mass',
               'cid_moissl_fat_free_mass_percent',
               'cid_moissl_fat_mass', 'cid_moissl_fat_mass_percent',
               'ecid_sbp', 'ecid_dbp', 'ecid_bpm']

blood_glucose_data = ['participant_id', 'CID', 'ecid_hba1c_percent', 'ecid_ldl',
                      'ecid_hdl', 'ecid_trig']

# Create DataFrames for each category of columns
intake_df = df[intake_variables]
anthro_df = df[anthro_data]
blood_glucose_df = df[blood_glucose_data]

# Assuming you have already created the 'anthro_df' DataFrame

# Define the original weights for each 'CID' value in a dictionary
original_weights = {
    1: 1/8,
    2: 1/8,
    3: 1/4,
    4: 1/2
}

# Get a list of unique 'participant_id' values
unique_participants = anthro_df['participant_id'].unique()

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['participant_id'] + ['Weighted Average for ' + col for col in anthro_df.columns[2:]])

# Iterate through each 'participant_id'
for target_participant in unique_participants:
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning for each participant
    anthro_df_copy = anthro_df.copy()
    
    # Check if the current 'participant_id' is present in the DataFrame
    if target_participant in anthro_df_copy['participant_id'].unique():
        print(target_participant)
        # Get the 'CID' values associated with the 'participant_id'
        valid_cids = anthro_df_copy[(anthro_df_copy['participant_id'] == target_participant) & anthro_df_copy[column].notna()]['CID'].unique()
        print(valid_cids)
        # Initialize a list to store weighted averages for each column of interest
        weighted_averages = [target_participant]
        
        for column in anthro_df.columns[2:]:
            # Calculate the weighted average for the current column and 'participant_id'
            weighted_average = 0
            
            # Check if there are valid 'CID' values
            if len(valid_cids) > 0:
                # Select only data where 'column' is not NaN
                valid_data = anthro_df_copy[
                    (anthro_df_copy['participant_id'] == target_participant) & 
                    anthro_df_copy[column].notna()
                ]
                
                if not valid_data.empty:
                    # Calculate the sum of original weights for the available 'CID' values
                    sum_original_weights = sum(original_weights[cid] for cid in valid_data['CID'])
                    
                    # Calculate the rescaled weights to ensure they add up to 1
                    valid_data['rescaled_weight'] = valid_data['CID'].map(original_weights) / sum_original_weights
                    
                    # Calculate the weighted average for the current column and 'participant_id'
                    weighted_average = (valid_data[column] * valid_data['rescaled_weight']).sum()
            
            weighted_averages.append(weighted_average)
        
        # Append the results to the result_df
        result_df = result_df.append(pd.Series(weighted_averages, index=result_df.columns), ignore_index=True)
    else:
        print(f"Target Participant {target_participant} is not present in the DataFrame.")

# Print the resulting DataFrame
print(result_df)

# Save the result as an Excel (.xlsx) file
result_df.to_excel('weighted_averages.xlsx', index=False)