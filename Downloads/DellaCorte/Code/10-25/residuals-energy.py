import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data from 'weighted_averages.xlsx' and specify the sheet
file_path = '/Users/user/Downloads/DellaCorte/Data/weighted_averages.xlsx'
sheet_name = 'Sqrt Mixed Weighted Averages'
data = pd.read_excel(file_path, sheet_name)

# Define the dietary variables you want to calculate residuals for
dietary_variables = data.columns[1:17]  # Assuming columns 1 to 16 are dietary variables

# Select the independent variable (square root of 'Weighted Average for intake_energy_kilocalories')
independent_variable = data['Sqrt of Weighted Average for intake_energy_kilocalories']

# Create an empty DataFrame to store the residuals
residuals_df = pd.DataFrame(columns=dietary_variables)

# Iterate through each dietary variable
for column_name in dietary_variables:
    # Extract the dependent variable (the dietary variable)
    dependent_variable = data[column_name]

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to predict the dependent variable using the independent variable
    model.fit(independent_variable.values.reshape(-1, 1), dependent_variable)

    # Calculate the residuals
    residuals = dependent_variable - model.predict(independent_variable.values.reshape(-1, 1))

    # Store the residuals in the DataFrame
    residuals_df[column_name] = residuals

# Rename the columns with 'Residuals of ' + column_name
residuals_df.columns = ['Residuals of ' + column for column in residuals_df.columns]

# Save the residuals DataFrame to a new Excel file
residuals_output_file = 'mixed_weights_residuals.xlsx'
residuals_df.to_excel(residuals_output_file, index=False)

# # Iterate through each dietary variable again to plot and save the plots
# for column_name in dietary_variables:
#     dependent_variable = data[column_name]
#     model = LinearRegression()
#     model.fit(independent_variable.values.reshape(-1, 1), dependent_variable)
#     residuals = dependent_variable - model.predict(independent_variable.values.reshape(-1, 1))

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

#     # # Plot the residuals
#     # axes[0].scatter(independent_variable, residuals, label='Residuals', color='b')
#     # axes[0].set_title(f'Residuals for {column_name}')
#     # axes[0].set_xlabel('Sqrt of Weighted Average for intake_energy_kilocalories')
#     # axes[0].set_ylabel('Residuals')

#     # # Plot the least-squares regression line
#     # predicted = model.predict(independent_variable.values.reshape(-1, 1))
#     # axes[1].scatter(independent_variable, dependent_variable, label='Data', color='g')
#     # axes[1].plot(independent_variable, predicted, label='Regression Line', color='r')
#     # axes[1].set_title(f'Regression for {column_name}')
#     # axes[1].set_xlabel('Sqrt of Weighted Average for intake_energy_kilocalories')
#     # axes[1].set_ylabel(column_name)

#     # plt.tight_layout()
    
#     # Save the current plot
#     plot_output_file = f'residuals_plot_{column_name}.png'
#     plt.savefig(plot_output_file)

