import pandas as pd

DATA_PATH = '/Users/user/Downloads/DellaCorte/Merged.xlsx'
sheet_name = 'Means_By_CID'

try:
    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
    print(df.columns)
except Exception as e:
    print("An error occurred:", e)
    
    
diabetes_col = 'ecid_diabetes_1'
participant_ids = 'participant_id'

# Find all participant_ids that have in df that 
# have a 1 in the diabetes_col 
# Filter the DataFrame to get only rows where 'ecid_diabetes_1' is equal to 1
diabetes_positive = df[df['ecid_diabetes_1'] == 1]

# Extract the 'participant_id' column from the filtered DataFrame
participant_ids_with_diabetes = diabetes_positive['participant_id']

# Print the result
print(participant_ids_with_diabetes)


# Load data from the second Excel file
DATA_PATH_2 = 'XSLX/Outcomes_with_BMR-10-09.xlsx'

try:
    df_2 = pd.read_excel(DATA_PATH_2)
except Exception as e:
    print("An error occurred while loading data from the second file:", e)

# Assuming you have the same 'ecid_diabetes_1' and 'participant_id' columns in the second DataFrame,
# you can filter it in a similar way

diabetes_col_2 = 'ecid_diabetes_1'
participant_ids_2 = 'participant_id'

# Filter the second DataFrame to get only rows where 'ecid_diabetes_1' is equal to 1
diabetes_positive_2 = df_2[df_2[diabetes_col_2] == 1]

# Extract the 'participant_id' column from the filtered DataFrame
participant_ids_with_diabetes_2 = diabetes_positive_2[participant_ids_2]

# Print the result for the second Excel file
print(participant_ids_with_diabetes_2)







