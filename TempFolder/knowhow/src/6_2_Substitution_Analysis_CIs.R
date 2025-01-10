# Load necessary libraries
library(rethinking)
library(dplyr)
library(readr)
library(tidyr)

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# List of file paths
file_paths <- list(
  "../results/substitutions/bodyfat_substitution_results.csv",
  "../results/substitutions/trig_substitution_results.csv",
  "../results/substitutions/hdl_substitution_results.csv",
  "../results/substitutions/ldl_substitution_results.csv",
  "../results/substitutions/hba1c_substitution_results.csv",
  "../results/substitutions/bmi_substitution_results.csv",
  "../results/substitutions/waistcirumference_substitution_results.csv"
)

# Initialize an empty dataframe to store the results
final_results <- data.frame(
  filename = character(),
  column_name = character(),
  mean = numeric(),
  ci_lower = numeric(),
  ci_upper = numeric(),
  stringsAsFactors = FALSE
)

# Function to calculate mean and CI for each column in a dataframe
calculate_mean_and_ci <- function(df, file_name) {
  result <- df %>%
    summarise(across(everything(), list(mean = ~ mean(.), ci = ~ PI(., prob = 0.95))))
  
  # Create a dataframe from the result
  result_long <- data.frame(
    column_name = rep(names(df), each = 2),
    stat = rep(c("mean", "ci"), times = ncol(df)),
    value = unlist(result)
  ) %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(filename = gsub("\\.csv", "", basename(file_name)),
           ci_lower = ci[1:ncol(df)],
           ci_upper = ci[(ncol(df)+1):(2*ncol(df))]) %>%
    select(filename, column_name, mean, ci_lower, ci_upper)
  
  return(result_long)
}

# Loop through each file, read the data, calculate the mean and CI, and store the results
for (file_path in file_paths) {
  # Load the dataset
  df <- read_csv(file_path)
  
  # Calculate mean and CI
  results <- calculate_mean_and_ci(df, file_path)
  
  # Append the results to the final dataframe
  final_results <- bind_rows(final_results, results)
}

# Write the final results to a CSV file
write_csv(final_results, "../results/substitutions/final_results_summary.csv")

# Print the final results
print(final_results)
