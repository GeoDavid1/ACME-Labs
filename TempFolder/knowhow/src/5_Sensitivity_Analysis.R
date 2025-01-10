# Load necessary libraries
library(rethinking)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the dataset
df_blood <- read.csv("../data/processed_files/subsetted_df_blood.csv")
df_anthro <- read.csv("../data/processed_files/subsetted_df_anthro.csv")

#-----------------------
#Sensitivity analysis reporting only the posteriors of beta predictors.

outcome_vars <- c('bodyfat_percent', 'ldl', 'hba1c_percent')
predictor_vars <- c('intake_nmes_kcal_std', 'intake_intrinsic_sugars_kcal_std')
covariates_to_exclude <- c("energy",
                           "intake_protein_kcal_std", 
                           "intake_fat_kcal_std", 
                           "intake_alcohol_kcal_std",
                           "intake_fiber_kcal_std"
)

energy_var <- c('intake_energy_kcal_std_nmes_protein',
                'intake_energy_kcal_std_nmes_fat',
                'intake_energy_kcal_std_nmes_alcohol',
                'intake_energy_kcal_std_intrinsic_sugar_protein',
                'intake_energy_kcal_std_intrinsic_sugar_fat',
                'intake_energy_kcal_std_intrinsic_sugar_alcohol')

# Prepare the results table
results_table <- data.frame(
  Outcome = character(),
  Predictor = character(),
  Covariate_Excluded = character(),
  Mean = numeric(),
  SD = numeric(),
  Lower_CI = numeric(),
  Upper_CI = numeric(),
  stringsAsFactors = FALSE
)

# Define a function to remove a covariate and fit the model
fit_model_excluding_covariate <- function(data, outcome_var, predictor_var, energy_var, covariate_to_exclude, anthro=TRUE) {
  # List of all covariates
  if (anthro) {
    covariates <- c("predictor", "intake_protein_kcal_std", "intake_alcohol_kcal_std","intake_fiber_kcal_std",
                    "intake_fat_kcal_std", "intake_fiber_kcal_std", "age", "energy")
  }
  else {
    covariates <- c("predictor", "intake_protein_kcal_std", "intake_alcohol_kcal_std","intake_fiber_kcal_std",
                    "intake_fat_kcal_std", "intake_fiber_kcal_std", "age", "energy", "bodyfat_percent_scale")
  }
  
  # Exclude the specified covariate if any
  if (!is.null(covariate_to_exclude)) {
    covariates <- setdiff(covariates, covariate_to_exclude)
  }
  
  # Construct the formula dynamically
  formula <- paste("mu <- a_gender[gender] + b_center[centre_id]", 
                   paste(paste0("b_", covariates, " * ", covariates), collapse = " + "),
                   sep = " + ")
  
  # Define priors for the remaining covariates
  priors <- paste(lapply(covariates, function(cov) paste0("b_", cov, " ~ dnorm(0, 1)")), collapse = ", ")
  
  # Combine everything into the alist
  model_alist <- eval(parse(text = paste0("alist(
      outcome ~ dnorm(mu, sigma),
      ", formula, ",
      a_gender[gender] ~ dnorm(0, 1),
      ", priors, ",
      b_center[centre_id] ~ dnorm(0, 1),
      sigma ~ dexp(1)
    )")))
  

  # Create the data object, removing the excluded covariate
  data_list <- list(
    outcome = data[[outcome_var]],
    predictor = data[[predictor_var]],
    energy = data[[energy_var]],
    intake_fiber_kcal_std = df$intake_fiber_kcal_std,
    intake_protein_kcal_std = data$intake_protein_kcal_std,
    intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
    centre_id = data$centre_id,
    gender = data$gender,
    intake_fat_kcal_std = data$intake_fat_kcal_std,
    age = data$age
  )
  
  # Conditionally add bodyfat_percent_scale if anthro is FALSE
  if (!anthro) {
    data_list$bodyfat_percent_scale <- data$bodyfat_percent_scale
  }
  
  # Remove the excluded covariate from the data object if any
  if (!is.null(covariate_to_exclude)) {
    data_list[[covariate_to_exclude]] <- NULL
  }
  
  # Convert the list back to a data frame
  data_df <- as.data.frame(data_list)
  
  # Fit the model
  fit <- quap(
    model_alist,
    data = data_df,
    control = list(maxit = 10000)
  )
  
  return(fit)
}

# Triple loop
results <- list()
for (outcome_var in outcome_vars) {
  anthro <- (outcome_var == 'bodyfat_percent')
  if (outcome_var != "bodyfat_percent"){
    df <- df_anthro  
  } else {
    df <- df_blood
  }
  
  for (predictor_var in predictor_vars) {
    # Baseline model without any exclusions
    energy_var_baseline <- switch(predictor_var,
                                  "intake_nmes_kcal_std" = 'intake_energy_kcal_std_nmes',
                                  "intake_intrinsic_sugars_kcal_std" = 'intake_energy_kcal_std_intrinsic_sugar')
    
    
    
    # Fit the baseline model
    fit_baseline <- fit_model_excluding_covariate(df, outcome_var, predictor_var, energy_var_baseline, NULL, anthro)
    
    # Store the baseline result
    results[[paste(outcome_var, predictor_var, "none", sep = "_")]] <- fit_baseline
    
    # Extract posterior summary using precis
    summary_baseline <- precis(fit_baseline,prob=.95)
    b_predictor_summary_baseline <- summary_baseline[rownames(summary_baseline) == "b_predictor", ]
    
    # Store the baseline result in the results table
    results_table <- rbind(results_table, data.frame(
      Outcome = outcome_var,
      Predictor = predictor_var,
      Covariate_Excluded = "none",
      Mean = b_predictor_summary_baseline["mean"],
      SD = b_predictor_summary_baseline["sd"],
      Lower_CI = b_predictor_summary_baseline["2.5%"],
      Upper_CI = b_predictor_summary_baseline["97.5%"]
    ))
    
    # Models with covariate exclusions
    covariates_to_exclude_here <- covariates_to_exclude
    if (outcome_var != "bodyfat_percent"){
      covariates_to_exclude_here <- c(covariates_to_exclude_here, "bodyfat_percent_scale")
    }
    
    for (covariate_to_exclude in covariates_to_exclude_here) {
      # Select the appropriate energy_var
      if (predictor_var == 'intake_nmes_kcal_std') {
        energy_var <- switch(covariate_to_exclude,
                             'intake_fiber_kcal_std' = 'intake_energy_kcal_std_nmes',
                             "bodyfat_percent_scale" = 'intake_energy_kcal_std_nmes',
                             "intake_protein_kcal_std" = 'intake_energy_kcal_std_nmes_protein',
                             "intake_fat_kcal_std" = 'intake_energy_kcal_std_nmes_fat',
                             "intake_alcohol_kcal_std" = 'intake_energy_kcal_std_nmes_alcohol',
                             "energy" = 'intake_energy_kcal_std_nmes' # Default if "energy" is excluded
        )
      } else if (predictor_var == 'intake_intrinsic_sugars_kcal_std') {
        energy_var <- switch(covariate_to_exclude,
                             "intake_fiber_kcal_std" = 'intake_energy_kcal_std_intrinsic_sugar',
                             "bodyfat_percent_scale" = 'intake_energy_kcal_std_intrinsic_sugar',
                             "intake_protein_kcal_std" = 'intake_energy_kcal_std_intrinsic_sugar_protein',
                             "intake_fat_kcal_std" = 'intake_energy_kcal_std_intrinsic_sugar_fat',
                             "intake_alcohol_kcal_std" = 'intake_energy_kcal_std_intrinsic_sugar_alcohol',
                             "energy" = 'intake_energy_kcal_std_intrinsic_sugar' # Default if "energy" is excluded
        )
      }
      
      # Fit the model
      fit <- fit_model_excluding_covariate(df, outcome_var, predictor_var, energy_var, covariate_to_exclude, anthro)
      
      # Store the result
      results[[paste(outcome_var, predictor_var, covariate_to_exclude, sep = "_")]] <- fit
      
      # Extract posterior summary using precis
      summary <- precis(fit,prob=.95)
      b_predictor_summary <- summary[rownames(summary) == "b_predictor", ]
      
      # Store the result in the results table
      results_table <- rbind(results_table, data.frame(
        Outcome = outcome_var,
        Predictor = predictor_var,
        Covariate_Excluded = covariate_to_exclude,
        Mean = b_predictor_summary["mean"],
        SD = b_predictor_summary["sd"],
        Lower_CI = b_predictor_summary["2.5%"],
        Upper_CI = b_predictor_summary["97.5%"]
      ))
    }
  }
}

#Write it out in superior fashion

# Function to format the summary into the desired string
format_summary <- function(summary_row) {
  mean <- round(summary_row["mean"], 2)
  lower_ci <- round(summary_row["2.5%"], 2)
  upper_ci <- round(summary_row["97.5%"], 2)
  return(paste(mean, "\n(", lower_ci, ", ", upper_ci, ")", sep = ""))
}

# Create a matrix for the results table
results_matrix <- matrix("", nrow = 8, ncol = 7)
colnames(results_matrix) <- c("Model", "BF %", "LDL", "Hba1c", "BF %", "LDL", "Hba1c")
rownames(results_matrix) <- c("","Causal", "Causal - energy", "Causal - protein", "Causal - satd fat", "Causal - alcohol", "Causal - bodyfat", "Causal - fiber ")

# Fill in the 'Model' column
results_matrix[, 1] <- rownames(results_matrix)

# Mapping of covariate exclusions to row indices
covariate_map <- list(
  "none" = 1,
  "energy" = 2,
  "intake_protein_kcal_std" = 3,
  "intake_fat_kcal_std" = 4,
  "intake_alcohol_kcal_std" = 5,
  "bodyfat_percent_scale" = 6,  
  "intake_fiber_kcal_std" = 7
)

# Populate the matrix with the formatted results
for (outcome_var in outcome_vars) {
  for (predictor_var in predictor_vars) {
    covariates_here <- c("none", covariates_to_exclude)
    if (outcome_var != 'bodyfat_percent'){
      covariates_here <- c(covariates_here, "bodyfat_percent_scale")
    }
    for (covariate in covariates_here) {
      key <- paste(outcome_var, predictor_var, covariate, sep = "_")
      fit <- results[[key]]
      summary <- precis(fit, prob = .95)
      b_predictor_summary <- summary[rownames(summary) == "b_predictor", ]
      formatted_summary <- format_summary(b_predictor_summary)
      
      row_index <- covariate_map[[covariate]]
      if (predictor_var == "intake_nmes_kcal_std") {
        col_index <- switch(outcome_var, "bodyfat_percent" = 2, "ldl" = 3, "hba1c_percent" = 4)
      } else if (predictor_var == "intake_intrinsic_sugars_kcal_std") {
        col_index <- switch(outcome_var, "bodyfat_percent" = 5, "ldl" = 6, "hba1c_percent" = 7)
      }
      
      results_matrix[row_index, col_index] <- formatted_summary
    }
  }
}

# Save the results matrix to a CSV file
write.csv(results_matrix, file = paste0("../results/sensitivity/sensitivity_outcome_beta_8x7.csv"), row.names = TRUE)


