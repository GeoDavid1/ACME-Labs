import pandas as pd

# Load data from the first Excel file
DATA_PATH = '/Users/user/Downloads/DellaCorte/XSLX/Merged.xlsx'
sheet_name = 'Means_By_CID'

df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)

# Define the participant_id and CID columns
participant_id_column = 'participant_id'
cid_column = 'CID'

# Count the number of unique CID values for each participant ID
participant_id_cid_counts = df.groupby(participant_id_column)[cid_column].nunique()

# Filter out participant IDs that have only CID == 1
excluded_participant_ids = participant_id_cid_counts[participant_id_cid_counts == 1].index
filtered_df = df[~df[participant_id_column].isin(excluded_participant_ids)]


# Print the excluded participant IDs
print("Excluded Participant IDs with CID == 1 and no other CIDs:")
print(excluded_participant_ids)
