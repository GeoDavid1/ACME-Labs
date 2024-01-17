import pandas as pd

# Define the path to your Excel file
data_file = 'Final_Data.xlsx'

# Load data from 'Final_Data.xlsx'
df = pd.read_excel(data_file)

# Now, 'df' contains your data from the Excel file 'Final_Data.xlsx'.

# Calculate the tertiles for the 'ecid_hba1c_percent' column
tertiles_ecid_hba1c = df['ecid_hba1c_percent'].quantile([0, 1/3, 2/3, 1])

# Define custom labels for tertiles
tertile_labels = ["1st", "2nd", "3rd"]

# Add a new column with custom labels for tertiles
df['ecid_hba1c_tertile'] = pd.cut(df['ecid_hba1c_percent'], bins=tertiles_ecid_hba1c, labels=tertile_labels)

print(df)

