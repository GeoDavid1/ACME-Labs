import pandas as pd
from scipy.stats import kruskal

# Load data from 'weighted_averages_with_tertiles.xlsx'
file_path = '/Users/user/Downloads/DellaCorte/Data/weighted_averages_with_tertiles.xlsx'
data = pd.read_excel(file_path)

# Specify the columns for which you want to perform the Kruskal-Wallis test
columns_to_test = data.columns.difference(['participant_id', 'Tertile intake_nmes', 'Tertile ecid_hba1c_percent'])

# Create an empty DataFrame to store the p-values for NMES tertiles
nmes_p_values = pd.DataFrame(columns=['Column', 'Kruskal_P_Value'])

# Create an empty DataFrame to store the p-values for Hba1c tertiles
hba1c_p_values = pd.DataFrame(columns=['Column', 'Kruskal_P_Value'])

# Iterate through each column and apply the Kruskal-Wallis test for NMES tertiles
for column in columns_to_test:
    tertile_data = [group.dropna() for tertile, group in data.groupby('Tertile intake_nmes')[column]]
    
    # Check variance within each group
    group_variances = [group.var() for group in tertile_data]
    
    # Check if there is variation within each group before performing the test
    if any(variance != 0 for variance in group_variances):
        # Perform the Kruskal-Wallis test
        try:
            stat, p_value = kruskal(*tertile_data)
        except ValueError:
            p_value = float('nan')
    else:
        p_value = float('nan')  # Set to NaN if there is no variation
        
    # Append the results to the NMES p-values DataFrame
    nmes_p_values = nmes_p_values.append({'Column': column, 'Kruskal_P_Value': p_value}, ignore_index=True)

# Iterate through each column and apply the Kruskal-Wallis test for Hba1c tertiles
for column in columns_to_test:
    tertile_data = [group.dropna() for tertile, group in data.groupby('Tertile ecid_hba1c_percent')[column]]
    
    # Check variance within each group
    group_variances = [group.var() for group in tertile_data]
    
    # Check if there is variation within each group before performing the test
    if any(variance != 0 for variance in group_variances):
        # Perform the Kruskal-Wallis test
        try:
            stat, p_value = kruskal(*tertile_data)
        except ValueError:
            p_value = float('nan')
    else:
        p_value = float('nan')  # Set to NaN if there is no variation
        
    # Append the results to the Hba1c p-values DataFrame
    hba1c_p_values = hba1c_p_values.append({'Column': column, 'Kruskal_P_Value': p_value}, ignore_index=True)

# Format the p-values as decimal instead of scientific notation
nmes_p_values['Kruskal_P_Value'] = nmes_p_values['Kruskal_P_Value'].apply(lambda x: f"{x:.20f}")
hba1c_p_values['Kruskal_P_Value'] = hba1c_p_values['Kruskal_P_Value'].apply(lambda x: f"{x:.20f}")

# Print and save the p-values for NMES tertiles
print("NMES Tertiles P-Values:")
print(nmes_p_values)

# Save the p-values for NMES tertiles to a text file
nmes_p_values.to_csv('/Users/user/Downloads/DellaCorte/nmes_kruskal_p_values.txt', sep='\t', index=False)

# Print and save the p-values for Hba1c tertiles
print("\nHba1c Tertiles P-Values:")
print(hba1c_p_values)

# Save the p-values for Hba1c tertiles to a text file
hba1c_p_values.to_csv('/Users/user/Downloads/DellaCorte/hba1c_kruskal_p_values.txt', sep='\t', index=False)





