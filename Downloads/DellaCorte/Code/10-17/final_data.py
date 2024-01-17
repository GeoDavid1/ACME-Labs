import pandas as pd
from scipy.stats import kruskal

# Read the 'Final_Data.xlsx' file into a DataFrame
final_data = pd.read_excel('/Users/user/Downloads/DellaCorte/Final_Data.xlsx')

# Get the columns you want to test (excluding 'NMES_Tertile')
columns_to_test = final_data.columns.difference(['NMES_Tertile'])

# Create an empty DataFrame to store the p-values
p_values = pd.DataFrame(columns=['Column', 'Kruskal_P_Value'])

# Iterate through each column and apply the Kruskal-Wallis test
for column in columns_to_test:
    tertile_data = [group.dropna() for tertile, group in final_data.groupby('NMES_Tertile')[column]]
    
    # Check variance within each group
    group_variances = [group.var() for group in tertile_data]
    
    # If all variances are not zero (i.e., there is variation in the group), perform the test
    if any(variance != 0 for variance in group_variances):
        # Perform the Kruskal-Wallis test
        stat, p_value = kruskal(*tertile_data)
    else:
        p_value = float('nan')  # Set to NaN if there is no variation
        
    # Append the results to the p_values DataFrame
    p_values = p_values.append({'Column': column, 'Kruskal_P_Value': p_value}, ignore_index=True)

# Print and save the p-values to a text file
print(p_values)

# Save the p-values to a text file
p_values.to_csv('kruskal_p_values.txt', sep='\t', index=False)

