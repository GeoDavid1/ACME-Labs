import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas library for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib library for plotting
# Import utility function for converting DataFrame to Excel rows
from openpyxl.utils.dataframe import dataframe_to_rows
# Import Alignment and Font classes from openpyxl for cell formatting
from openpyxl.styles import Font
import re
import seaborn as sns
from scipy import stats

import statsmodels.api as sm


def calculate_lsmeans_basic(df, predictor_columns, outcome_column):
    """
    Calculates the least square means for each predictor of a given dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    predictor_columns (list): A list of predictor column names.
    outcome_column (str): The name of the outcome column.

    Returns:
    dict: A dictionary containing the mean, lower confidence interval, and upper confidence interval
          for each predictor.
    """
    # Fit the model on the whole dataset
    formula = f"{outcome_column} ~ {' + '.join(predictor_columns)}"
    model = sm.formula.ols(formula=formula, data=df).fit()

    results_dict = {}
    for predictor in predictor_columns:
        # Get the predicted values for the predictor
        predictions = model.get_prediction(df[predictor]).summary_frame()

        # Calculate the LSMeans and confidence intervals for the predictor
        mean = predictions['mean'].mean()
        lci = predictions['mean_ci_lower'].mean()
        uci = predictions['mean_ci_upper'].mean()

        results_dict[predictor] = [mean, lci, uci]

    return results_dict


def calculate_lsmeans(df, predictor_columns, outcome_column):
    """
    Calculates the least square means for each tertile of a given dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    predictor_columns (list): A list of predictor column names.
    outcome_column (str): The name of the outcome column.

    Returns:
    dict: A dictionary containing the mean, lower confidence interval, and upper confidence interval
          for each tertile.
    """
    # Fit the model on the whole dataset
    formula = f"{outcome_column} ~ {' + '.join(predictor_columns)}"
    model = sm.formula.ols(formula=formula, data=df).fit()

    results_dict = {}
    for tertile in ['Low', 'Medium', 'High']:
        # Create a subset for each tertile
        df_subset = df[df['tertile'] == tertile].copy()

        # Get the predicted values for the subset
        predictions = model.get_prediction(
            df_subset[predictor_columns]).summary_frame()

        # Calculate the LSMeans and confidence intervals for the subset
        mean = predictions['mean'].mean()
        lci = predictions['mean_ci_lower'].mean()
        uci = predictions['mean_ci_upper'].mean()

        results_dict[tertile] = [mean, lci, uci]

    return results_dict


def visualize_transformations(df, var_dict):
    """
    Visualizes the transformations of variables in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        var_dict (dict): A dictionary mapping variable categories to a list of variables.

    Returns:
        None
    """

    # Create subplots for each variable
    fig, ax = plt.subplots(50, 2, figsize=(10, 5*50))
    ctr = 0
    ax = ax.flatten()

    # Iterate over variable categories
    for key in var_dict:
        # Iterate over variables in each category
        for var in var_dict[key]:
            # Plot histogram of residuals if available, else plot histogram of variable
            if var+'_residuals' in df.columns:
                ax[ctr].hist(df[var+'_residuals'], bins=100)
                ax[ctr].set_title('Residual '+var)
            else:
                ax[ctr].hist(df[var], bins=100)
                ax[ctr].set_title(var)

            ctr += 1

            # Plot histogram of normalized residuals if available, else plot histogram of power transformed variable
            if var+'_normed_residuals' in df.columns:
                ax[ctr].hist(df[var+'_normed_residuals'], bins=100)
                ax[ctr].set_title('Power Transform Residual', fontsize=10)
            else:
                ax[ctr].hist(get_power_transform(df[var].to_numpy()), bins=100)
                ax[ctr].set_title('Power Transform')

            ctr += 1

    # Save the figure
    plt.savefig('../data/results/figures/transformations.pdf',
                dpi=600, bbox_inches='tight')
    plt.close()


def recursive_cum_avg(list_of_measurements):
    '''
    Calculate the cumulative average of a list of measurements using a recursive approach.
    Recursive implementation to deal with any number of CIDS - just sort them by date (most recent measure first)

    Parameters:
    list_of_measurements (list): A list of measurements.

    Returns:
    float: The cumulative average of the measurements.

    Raises:
    TypeError: If the input is not a list.
    ValueError: If the input list is empty.

    Example:
    >>> recursive_cum_avg([3, 5, 5])
    4.375
    >>> recursive_cum_avg([5, 4, 5, 2])
    4.625
    >>> recursive_cum_avg([5, 4, 3])
    4.0
    >>> recursive_cum_avg([3])
    3.0
    >>> recursive_cum_avg([0])
    0.0
    >>> recursive_cum_avg([])
    ValueError: Provided empty list
    >>> recursive_cum_avg("its a bug")
    TypeError: Wrong input, needs a list
    '''

    if type(list_of_measurements) != list:
        try:
            list_of_measurements = list_of_measurements.to_list()
        except:
            raise TypeError('Wrong input, needs a list')

    list_of_measurements = np.array(list_of_measurements)
    idx = np.where(list_of_measurements > 0)
    list_of_measurements = list_of_measurements[idx]

    if len(list_of_measurements) == 0:
        return 0

    if len(list_of_measurements) == 1:
        return list_of_measurements[0]

    def recursion_helper(prev_avg, list_of_measurements):
        next_measurement = list_of_measurements[0]
        next_average = (prev_avg + next_measurement) / 2
        if len(list_of_measurements) == 1:
            return next_average
        else:
            list_of_measurements = list_of_measurements[1:]
            return recursion_helper(next_average, list_of_measurements)

    return recursion_helper(list_of_measurements[0], list_of_measurements[1:])


def get_bmi(weight, height):
    """
    Calculate the Body Mass Index (BMI) for adults.

    Parameters:
    - weight (float): Body weight in pounds.
    - height (float): Body height in inches.

    Returns:
    - float: The calculated BMI value.

    Formula:
    BMI = (weight / (height ** 2)) * 703

    Note:
    - The BMI is a measure of body fat based on weight and height.
    - It is commonly used to classify individuals into categories such as underweight, normal weight, overweight, and obese.
    """
    bmi_value = (weight / (height ** 2)) * 703

    return bmi_value


def calculate_bmi(weight_kilos, height_centimeters):
    """
    Calculate BMI (Body Mass Index) using weight in kilograms and height in centimeters.

    Parameters:
    - weight_kilos (float): Weight in kilograms.
    - height_centimeters (float): Height in centimeters.

    Returns:
    float: Calculated BMI.
    """
    weight_pounds = weight_kilos * 2.20462
    height_inches = height_centimeters * 0.393701
    bmi = get_bmi(weight_pounds, height_inches)
    return bmi


def calculate_multivariate_regression(df: pd.DataFrame, outcome_column: str, input_columns: list) -> dict:
    """
    Calculate multivariate regression on one outcome column from a pandas dataframe
    using multiple input columns from the same dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    outcome_column (str): The name of the outcome column.
    input_columns (list): The list of input column names.

    Returns:
    dict: A dictionary containing the p-values and slopes (beta values) for each input column.
    """

    # Add a constant column to the input dataframe
    # List of relevant columns
    relevant_columns = [outcome_column] + input_columns

    # Replace inf and -inf with nan
    df[relevant_columns] = df[relevant_columns].replace(
        [np.inf, -np.inf], np.nan)

    # Drop rows with nan
    df = df.dropna(subset=relevant_columns)

    X = sm.add_constant(df[input_columns])

    # Fit the OLS model
    model = sm.OLS(df[outcome_column], X).fit()

    # Get the p-values and slopes (beta values) for each input column
    p_values = model.pvalues[1:]  # Exclude the constant column
    slopes = model.params[1:]  # Exclude the constant column

    # Create a dictionary to store the results
    results = {}
    for i, column in enumerate(input_columns):
        results[column] = {'p-value': p_values[i], 'slope': slopes[i]}

    return results


def get_power_transform(residuals):
    """
    Apply power transform to normalize the residuals.

    Parameters:
    residuals (array-like): The residuals to be transformed.

    Returns:
    array-like: The transformed residuals.
    """

    # Initialize the PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')

    # Fit and transform the residuals
    return (pt.fit_transform(residuals.reshape(-1, 1)))


def calculate_residuals(df: pd.DataFrame, reference: str, variable: str) -> np.ndarray:
    """
    Calculate residuals for a given variable using linear regression after normalizing,
    handling NaN values.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - reference (str): The name of the reference column.
    - variable (str): The name of the variable column.

    Returns:
    - np.ndarray: An array of residuals with the same shape as the original variable column.
    """

    # Drop rows with NaN values in either reference or variable columns
    df_clean = df.dropna(subset=[reference, variable])

    # Check if there are any rows remaining after dropping NaN values
    if df_clean.shape[0] == 0:
        raise ValueError("No valid data points after dropping NaN values.")

    # Reshape the input features and target variable
    X = df_clean[reference].values.reshape(-1, 1)
    y = df_clean[variable].values.reshape(-1, 1)

    # Fit a linear regression model
    regressor = LinearRegression()
    regressor.fit(X, y)

    # Calculate the residuals for all data
    residuals = y - regressor.predict(X)

    # Create an array of NaN values with the same shape as the original variable column
    nan_residuals = np.full_like(df[variable].values, np.nan)

    # Replace the corresponding positions with the calculated residuals
    nan_residuals[df_clean.index] = residuals.flatten()

    return nan_residuals


def generate_nutrient_levels_table(final_data, excel_filename="../data/results/tables/nutrient_levels_table.xlsx"):
    """
    Generate a nutrient levels table from the given final data and save it to an Excel file and a PDF file.

    Parameters:
    - final_data (pd.DataFrame): The final data containing information about predictors, outcomes, models, tertiles, and P trends.
    - excel_filename (str): The filename to save the Excel file (default: "nutrient_levels_excel.xlsx").
    - pdf_filename (str): The filename to save the PDF file (default: "nutrient_levels_matplotlib.pdf").

    Returns:
    None

    The function creates a table with information about predictors, outcomes, models, tertiles, and P trends.
    It saves the table as an Excel file and generates a PDF file with a visual representation of the table using matplotlib.

    Example usage:
    generate_nutrient_levels_table(final_data)
    """
    # Unique values of predictors, outcomes, and models
    predictors = final_data['Predictor'].unique()
    outcomes = final_data['Outcome'].unique()

    # Create an empty table with headers for predictors
    predictor_headers = [""] * 2

    for predictor in predictors:
        # Add predictor headers with formatting
        predictor_headers.append(f"{predictor}{' ' * 3}")
        predictor_headers.extend([""] * 3)

    data = [predictor_headers]
    data.append([" "] + ["Low (T1)", "Moderate (T2)",
                "High (T3)", "P"] * len(predictors))

    # Add rows dynamically based on unique values of outcomes and models
    for outcome in outcomes:
        models_for_outcome = final_data[final_data['Outcome']
                                        == outcome]['Model'].unique()

        # Add a space row before each outcome_row
        data.append([""] * len(data[0]))

        # Create and add the outcome_row
        if pd.notna(outcome):
            outcome_row = [""] * (len(data[0]) // 2) + [f" {outcome} "] + [
                ""] * (len(data[0]) - len(data[0]) // 2 - 1)
            data.append(outcome_row)

        if pd.isna(outcome):
            # For outcome = NaN and model = NaN, create a model called "Intake" to consolidate data
            if pd.isna(final_data['Model'].iloc[0]) or final_data['Model'].iloc[0] == "":
                models_for_outcome = ["Intake"]

                # Create a single row for "Intake" model with combined tertiles and P values
                intake_row = ["Intake"]
                for predictor in predictors:
                    for level in ["Low (T1)", "Moderate (T2)", "High (T3)", "P trend"]:
                        filtered_data = final_data[
                            (final_data['Predictor'] == predictor) &
                            (final_data['Outcome'].isna()) &
                            (final_data['Model'].isna())
                        ]
                        # Check for NaN values and handle appropriately
                        value = filtered_data[level].values[0] if not filtered_data.empty and pd.notna(
                            filtered_data[level].values[0]) else "N/A"
                        intake_row.append(value)
                data.append(intake_row)

        for model in models_for_outcome:
            row_data = [f"{model}"]
            for predictor in predictors:
                for level in ["Low (T1)", "Moderate (T2)", "High (T3)", "P trend"]:
                    filtered_data = final_data[
                        (final_data['Predictor'] == predictor) &
                        (final_data['Outcome'] == outcome) &
                        (final_data['Model'] == model)
                    ]
                    # Check for NaN values and handle appropriately
                    value = filtered_data[level].values[0] if not filtered_data.empty and pd.notna(
                        filtered_data[level].values[0]) else "N/A"
                    row_data.append(value)
            data.append(row_data)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Drop rows with NaN or blank values
    df = df.dropna(how='all').fillna("")

    # Remove the second intake row
    df = df[~((df[0] == "Intake") & (df[1].str.contains("N/A")))]

    # Remove the "nan" from the first row
    df.iloc[3] = df.iloc[3].replace("nan", "")

    # Create an Excel writer using openpyxl
    with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='w') as writer:
        # Write the DataFrame to the Excel file
        df.to_excel(writer, index=False, header=False, sheet_name='Sheet1')

        # Access the openpyxl workbook and worksheet
        workbook = writer.book
        worksheet = workbook['Sheet1']

        # Iterate through all columns and set the width to the max length of data in the column
        for column in worksheet.columns:
            max_length = 0
            column = [column[0]] + [cell for cell in column[1:]]
            for cell in column:
                try:  # Necessary to avoid error on empty cells
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column[0]
                                        .column_letter].width = adjusted_width

        # Bold the first two rows
        for row in worksheet.iter_rows(min_row=1, max_row=2):
            for cell in row:
                cell.font = Font(bold=True)


def generate_characteristics_table(df, file_path="../data/results/tables/characteristics_table.xlsx"):

    # List of variables categorized by data type
    variables = {
        'Age': ['age'],
        'Dietary Data': ['intake_energy_mj', 'intake_fat', 'intake_protein', 'intake_fibre_englyst',
                         'intake_carbohydrate', 'intake_total_sugars', 'intake_sucrose', 'intake_fructose',
                         'intake_glucose', 'intake_nmes', 'intake_intrinsic_sugars', 'intake_starch', 'intake_cholesterol', 'intake_alcohol', 'intake_satd_fa'],
        'Anthropometric Data': ['bmi', 'bodyfat', 'bodyfat_percent', 'waistcirumference'],
        'Blood Data': ['hba1c', 'hba1c_percent', 'ldl', 'hdl', 'trig']
    }

    # Drop participants with any null values for any label
    for cols in variables.values():
        df.dropna(subset=cols, inplace=True)

    # Define the variables that will be counted as carbohydrates for energy intake analysis
    carbohydrate_variables = ['intake_carbohydrate', 'intake_total_sugars', 'intake_sucrose', 'intake_fructose',
                              'intake_glucose', 'intake_nmes', 'intake_intrinsic_sugars', 'intake_starch']
    # Fats
    fat_variables = ['intake_fat', 'intake_satd_fa']
    # Proteins
    protein_variables = ['intake_protein']

    # Variables that are on the characteristic tables as percentiles
    percentile_variables = ['age', 'intake_energy_mj', 'intake_fibre_englyst', 'intake_cholesterol',
                            'intake_alcohol', 'bmi', 'bodyfat', 'bodyfat_percent', 'waistcirumference', 'hba1c', 'hba1c_percent', 'ldl', 'hdl', 'trig']

    # Variables that are on the characteristic tables as standard error
    se_variables = ['intake_fat', 'intake_protein', 'intake_carbohydrate', 'intake_total_sugars', 'intake_sucrose',
                    'intake_fructose', 'intake_glucose', 'intake_nmes', 'intake_intrinsic_sugars', 'intake_starch', 'intake_satd_fa']

    # Calculate total energy for normalization
    TOTAL_ENERGY = df['intake_energy_mj'].mean()

    # Create a DataFrame to store the information
    table_data = pd.DataFrame(columns=['Variable', 'N', 'Mean'])

    # Populate the DataFrame with information about each variable
    for category, variables_list in variables.items():
        # Check if the category is "Age" and skip adding the label in that case
        if category != 'Age':
            table_data = pd.concat([table_data, pd.DataFrame({'Variable': [None], 'N': [''], 'Mean': [
                                   '']}), pd.DataFrame({'Variable': [category], 'N': [''], 'Mean': ['']})], ignore_index=True)

        for variable in variables_list:
            # Capitalize each word in the variable name
            variable_label = ' '.join([word.capitalize()
                                      for word in variable.split('_')])

            # Count non-null values and calculate the mean for the current variable
            num_values = df[variable].count()

            # For percentile variables, get the 25th and 75th percentiles and put in the 'Median' column as 'median (25th_percentile, 75th_percentile)'
            if variable in percentile_variables:
                # Get the mean
                mean_value = df[variable].mean()

                # Calculate the standard deviation
                std_deviation_value = np.std(
                    df[variable], ddof=1) / np.sqrt(num_values - 1)

                # Calculate the confidence interval (mean ± t * standard deviation), where t is the t-statistic for the desired confidence level
                # for a 95% confidence interval
                t_statistic = stats.t.ppf(0.975, df=num_values - 1)
                margin_of_error = t_statistic * std_deviation_value
                lower_bound = mean_value - margin_of_error
                upper_bound = mean_value + margin_of_error

                # Format 'Mean' column as 'mean (t * SE)' for the confidence interval
                average = f"{mean_value:.2f} ({lower_bound:.2f}, {upper_bound:.2f})"

            # For se_variables, get the mean energy_intake_percentage and the standard error and output in 'Mean' column as 'mean ± standard error'
            elif variable in se_variables:
                if variable in carbohydrate_variables or variable in protein_variables:
                    energy_intakes = df[variable] * 17 / 1000
                    energy_intake_percentages = energy_intakes * 100 / TOTAL_ENERGY
                elif variable in fat_variables:
                    energy_intakes = df[variable] * 37 / 1000
                    energy_intake_percentages = energy_intakes * 100 / TOTAL_ENERGY

                mean_value = energy_intake_percentages.mean()

                # Use numpy to calculate the true standard error
                std_deviation_value = np.std(
                    energy_intake_percentages, ddof=1) / np.sqrt(num_values - 1)

                # Calculate the confidence interval (mean +- t * standard deviation), where t is the t-statistic for the desired confidence level
                # for a 95% confidence interval
                t_statistic = stats.t.ppf(0.975, df=num_values - 1)
                margin_of_error = t_statistic * std_deviation_value
                lower_bound = mean_value - margin_of_error
                upper_bound = mean_value + margin_of_error

                # Format 'Mean' column as 'mean ± STD'
                average = f"{mean_value:.2f} ({lower_bound:.2f}, {upper_bound:.2f})"

            # Concatenate information for the current variable to the DataFrame
            table_data = pd.concat([table_data, pd.DataFrame({'Variable': [
                                   variable_label], 'N': [num_values], 'Mean': [average]})], ignore_index=True)

    # Create an Excel writer with openpyxl
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Write the DataFrame to the Excel file
        table_data.to_excel(writer, index=False, sheet_name='Sheet1')

        # Access the Excel workbook and sheet
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Set the font style to bold for category names
        bold_font = Font(bold=True)

        # Iterate through categories and bold the names
        for category, variables_list in variables.items():
            # Check if the category is "Age" and skip bolding in that case
            if category != 'Age':
                # Find the row number where the category name is located
                matching_rows = table_data[table_data['Variable'] == category]

                if not matching_rows.empty:
                    # Add 2 for 0-based index and header row
                    category_row = matching_rows.index[0] + 2

                    # Apply bold font to the entire row
                    for col_num in range(1, len(table_data.columns) + 1):
                        worksheet.cell(row=category_row,
                                       column=col_num).font = bold_font

    return table_data


def check_condition(df):
    """
    Check if the mean is greater than the first tertile and whether the first tertile is greater than the second tertile.

    Parameters:
    - df (DataFrame): DataFrame containing the data.

    Returns:
    """
    # Get column names from the DataFrame
    columns = df.columns
    # Iterate over every column
    for i in range(0, len(columns)):
        # Iterate over rows in the DataFrame
        for index, row in df.iterrows():
            # Extract values from the parentheses using string manipulation
            values_str = str(row[columns[i]])
            # Split the string to extract mean and tertile values
            mean, tertile_values = values_str.split(' (')
            # Extract and convert values for comparison
            first_tertile, second_tertile = map(
                float, tertile_values[:-1].split(';'))

            # Check the conditions
            if float(mean) <= first_tertile or float(mean) >= second_tertile:
                # Print information when conditions are not met
                print(f"Condition not met for {columns[i]} at index {index}")


def append_results_to_target_df(target_df, results_lsm, predictor, outcome, p_value, modelname='Baseline'):
    """
    Appends the results of a linear regression analysis to a target DataFrame.

    Parameters:
    target_df (pd.DataFrame): The target DataFrame to append the results to.
    results_lsm (dict): The results of the linear regression analysis.
    predictor (str): The name of the predictor variable.
    outcome (str): The name of the outcome variable.
    p_value (float): The p-value of the linear regression analysis.
    modelname (str, optional): The name of the model. Defaults to 'Baseline'.

    Returns:
    pd.DataFrame: The target DataFrame with the appended results.
    """

    # Check if p-value is less than 0.05
    if p_value < 0.0001:
        # If p-value is very small, set it to '<0.0001'
        p_value = '<0.0001'
    else:
        # If p-value is not very small, round it to 5 decimal places
        p_value = np.round(p_value, 5)

    # Create a dictionary to store the results
    results = {}
    results['Predictor'] = predictor
    results['Outcome'] = outcome
    results['Model'] = modelname

    # Format and store the results for different categories (Low, Moderate, High)
    results['Low (T1)'] = str(np.round(results_lsm['Low'][0], 2)) + ' (' + str(np.round(
        results_lsm['Low'][1], 2)) + '; ' + str(np.round(results_lsm['Low'][2], 2)) + ')'
    results['Moderate (T2)'] = str(np.round(results_lsm['Medium'][0], 2)) + ' (' + str(np.round(
        results_lsm['Medium'][1], 2)) + '; ' + str(np.round(results_lsm['Medium'][2], 2)) + ')'
    results['High (T3)'] = str(np.round(results_lsm['High'][0], 2)) + ' (' + str(np.round(
        results_lsm['High'][1], 2)) + '; ' + str(np.round(results_lsm['High'][2], 2)) + ')'

    # Store the p-value result
    results['P trend'] = p_value

    # Append the results to target_df
    target_df = pd.concat([target_df, pd.DataFrame(
        results, index=[0])], ignore_index=True)

    # Return the updated target_df
    return target_df


def process_predictor_outcomes(var_dict, transformed_df, raw_df):
    """
    Process predictor outcomes for given variable dictionaries, transformed data, and raw data.

    Parameters:
    - var_dict: Dictionary containing variable types and configurations.
    - transformed_df: DataFrame with transformed variables.
    - raw_df: DataFrame with raw data.

    Returns:
    - target_df: DataFrame containing processed predictor outcomes.
    """

    # Initialize target DataFrame to store results
    target_df = pd.DataFrame()

    # Loop through each predictor variable
    for predictor in var_dict['predictors']:
        # Assign tertiles to raw data
        transformed_df['tertile'] = pd.qcut(
            transformed_df[predictor], q=3, labels=['Low', 'Medium', 'High'])
        lookup_dict = dict(
            zip(transformed_df['Participant_ID'], transformed_df['tertile']))
        raw_df['tertile'] = raw_df['Participant_ID'].map(lookup_dict)

        # Calculate means and 95% CIs for each tertile
        mean_ci_df = raw_df.groupby('tertile')[predictor].agg([
            'mean', 'count', 'std'])
        mean_ci_df['lower_ci'] = mean_ci_df['mean'] - 1.96 * \
            (mean_ci_df['std'] / np.sqrt(mean_ci_df['count']))
        mean_ci_df['upper_ci'] = mean_ci_df['mean'] + 1.96 * \
            (mean_ci_df['std'] / np.sqrt(mean_ci_df['count']))

        # Define different model configurations
        baseline_model = var_dict['baseline_model'] + [predictor]
        anthro_model = var_dict['anthro_model'] + [predictor]
        blood_model = var_dict['blood_model'] + [predictor]

        # Create a dictionary to store the results for the current predictor
        results = {
            'Predictor': predictor,
            'Outcome': pd.NA,
            'Model': pd.NA,
            'Low (T1)': f"{np.round(mean_ci_df['mean']['Low'], 2)} ({np.round(mean_ci_df['lower_ci']['Low'], 2)}; {np.round(mean_ci_df['upper_ci']['Low'], 2)})",
            'Moderate (T2)': f"{np.round(mean_ci_df['mean']['Medium'], 2)} ({np.round(mean_ci_df['lower_ci']['Medium'], 2)}; {np.round(mean_ci_df['upper_ci']['Medium'], 2)})",
            'High (T3)': f"{np.round(mean_ci_df['mean']['High'], 2)} ({np.round(mean_ci_df['lower_ci']['High'], 2)}; {np.round(mean_ci_df['upper_ci']['High'], 2)})",
            'P trend': pd.NA
        }

        # Append the results to target_df
        target_df = pd.concat([target_df, pd.DataFrame(
            results, index=[0])], ignore_index=True)

        # Loop through outcome variables in 'anthro' category
        for outcome in var_dict['anthro']:
            analysis_df = transformed_df.copy()
            # Replace the transformed outcome with the raw outcome
            analysis_df = analysis_df.merge(
                raw_df[['Participant_ID', outcome]], on='Participant_ID', how='left')
            analysis_df.drop(outcome+'_x', axis=1, inplace=True)
            analysis_df.rename(columns={outcome+'_y': outcome}, inplace=True)

            # Run the model and calculate results
            results_lsm = calculate_lsmeans(
                analysis_df, baseline_model, outcome)
            p_data = calculate_multivariate_regression(
                transformed_df, outcome, baseline_model)
            p_value = p_data[predictor]['p-value']

            # Append the results to target_df
            target_df = append_results_to_target_df(
                target_df, results_lsm, predictor, outcome, p_value)

            # Repeat the process for the 'anthro' model
            results_lsm = calculate_lsmeans(analysis_df, anthro_model, outcome)
            p_data = calculate_multivariate_regression(
                transformed_df, outcome, anthro_model)
            p_value = p_data[predictor]['p-value']
            target_df = append_results_to_target_df(
                target_df, results_lsm, predictor, outcome, p_value, modelname='anthro')

        # Loop through outcome variables in 'blood' category
        for outcome in var_dict['blood']:
            analysis_df = transformed_df.copy()
            analysis_df = analysis_df.merge(
                raw_df[['Participant_ID', outcome]], on='Participant_ID', how='left')
            analysis_df.drop(outcome+'_x', axis=1, inplace=True)
            analysis_df.rename(columns={outcome+'_y': outcome}, inplace=True)

            results_lsm = calculate_lsmeans(
                analysis_df, baseline_model, outcome)
            p_data = calculate_multivariate_regression(
                transformed_df, outcome, baseline_model)
            p_value = p_data[predictor]['p-value']

            # Append the results to target_df
            target_df = append_results_to_target_df(
                target_df, results_lsm, predictor, outcome, p_value)

            # Repeat the process for the 'blood' model
            results_lsm = calculate_lsmeans(analysis_df, blood_model, outcome)
            p_data = calculate_multivariate_regression(
                transformed_df, outcome, blood_model)
            p_value = p_data[predictor]['p-value']

            target_df = append_results_to_target_df(
                target_df, results_lsm, predictor, outcome, p_value, modelname='blood')

    return target_df


def parse_values(value_str):
    # Use regular expressions to extract the values
    match = re.match(r'([\d.]+) \(([\d.]+); ([\d.]+)\)', value_str)

    if match:
        # Extract the values from the match object
        median = float(match.group(1))
        low_err = float(match.group(2))
        upper_err = float(match.group(3))

        return median, low_err, upper_err
    else:
        # Return None if the string doesn't match the expected format
        return None


def plot_nutrient_levels(file_path='../data/results/tables/nutrient_levels_table.xlsx', save_path='../data/results/figures/nutrient_levels_plot.png'):
    # Read the Excel file into a DataFrame without skipping the first row
    df_complete = pd.read_excel(file_path)
    predictors = df_complete.columns[df_complete.columns.str.startswith(
        'Unnamed:') == False].unique()
    outcomes = ['bmi', 'bodyfat_percent', 'waistcirumference',
                'ldl', 'hdl', 'trig', 'hba1c_percent']
    anthro = ['bmi', 'bodyfat_percent', 'waistcirumference']

    # Drop rows with NaN values
    df_complete = df_complete.dropna()

    # Get the dimensions of the DataFrame
    M, N = df_complete.shape

    tertile_blocks = []
    p_value_blocks = []
    # Iterate over the range of rows
    for i in range(1, N):
        # Check if the current row index meets the condition
        if i % 4 == 2:
            # Iterate over the range of columns
            for j in range(2, M):
                # Check if the current column index meets the condition
                if j % 2 == 0:
                    # Select the relevant columns using iloc
                    selected_columns = df_complete.iloc[j-1:j+1, i-1:i+2]
                    p_values = pd.to_numeric(
                        df_complete.iloc[j-1:j+1, i+2], errors='coerce').round(2)

                    tertile_blocks.append(selected_columns)
                    p_value_blocks.append(p_values)

    # Set seaborn style
    sns.set(style="whitegrid")

    # Create subplot grid
    fig, axs = plt.subplots(6, 7, figsize=(70, 60))

    anthro_bool = False
    for i, (tertile_block, p_value_block) in enumerate(zip(tertile_blocks, p_value_blocks)):
        pred_indx = i // 7
        outcome_indx = i % 7
        predictor = predictors[pred_indx]
        outcome = outcomes[outcome_indx]
        anthro_bool = outcome in anthro
        if anthro_bool:
            other = "Anthro"
        else:
            other = "Blood"
        base_model = tertile_block.iloc[0]
        other_model = tertile_block.iloc[1]
        # Create arbitrary x-axis positions
        x_positions = np.array([1, 3, 5, 7, 9, 11])
        label_positions = np.array([2, 6, 10])
        # Name the x_positions at 2, 6, 10 as 'T1', 'T2', 'T3' on the graph
        for k, base_val in enumerate(base_model):
            median, low_err, upper_err = parse_values(base_val)
            if k == 0:
                axs[pred_indx, outcome_indx].errorbar(x_positions[2 * k], median, yerr=[[median - low_err], [
                                                      upper_err - median]], fmt='o', color='blue', label=f'Basic Model, P-value: {p_value_block.iloc[0]}', capsize=10)
            else:
                axs[pred_indx, outcome_indx].errorbar(x_positions[2 * k], median, yerr=[
                                                      [median - low_err], [upper_err - median]], fmt='o', color='blue', capsize=10)

        for l, other_val in enumerate(other_model):
            median, low_err, upper_err = parse_values(other_val)
            if l == 0:
                axs[pred_indx, outcome_indx].errorbar(x_positions[2 * l + 1], median, yerr=[[median - low_err], [
                                                      upper_err - median]], fmt='o', color='red', label=f'{other} Model, P-value: {p_value_block.iloc[1]}', capsize=10)
            else:
                axs[pred_indx, outcome_indx].errorbar(x_positions[2 * l + 1], median, yerr=[
                                                      [median - low_err], [upper_err - median]], fmt='o', color='red', capsize=10)

        # Set x-axis labels
        axs[pred_indx, outcome_indx].set_xticks(label_positions)
        axs[pred_indx, outcome_indx].set_xticklabels(['T1', 'T2', 'T3'])
        axs[pred_indx, outcome_indx].set_title(
            f'Basic/{other} Model: {predictor}/{outcome}')
        axs[pred_indx, outcome_indx].set_xlabel('Tertile')
        axs[pred_indx, outcome_indx].set_ylabel('Nutrient Amount')
        axs[pred_indx, outcome_indx].legend(loc='upper left')
        axs[pred_indx, outcome_indx].grid(False)
        axs[pred_indx, outcome_indx].spines['top'].set_visible(False)
        axs[pred_indx, outcome_indx].spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()
