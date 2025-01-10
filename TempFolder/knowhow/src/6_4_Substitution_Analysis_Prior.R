# Load necessary libraries
library(rethinking)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the dataset
df <- read.csv("../data/processed_files/subsetted_df.csv")

# Define change amounts
change <- 40  # kcal; 12.5 g = 50 kcal
change_fiber <- 10  # kcal ; 12.5 g = 25 kcal

# Function to calculate the log-transformed standardized change
log_std_change <- function(mean, change, mean_log, sd_log) {
  log_change <- log(mean + change) - mean_log
  std_change <- log_change / sd_log
  return(std_change)
}

# Main function to calculate the change in outcome
calculate_change <- function(model, predictor, substitution, change, change_sub) {
  # Means and standard deviations of the log-transformed predictors
  log_means <- c(
    predictor = mean(log(predictor)), 
    substitute = mean(log(substitution))
  )
  log_sds <- c(
    predictor = sd(log(predictor)), 
    substitute = sd(log(substitution))
  )
  
  # Example values of original predictor and substitute
  mean_predictor <- mean(predictor)
  mean_substitution <- mean(substitution)
  
  # Calculate standardized changes
  delta_predictor <- log_std_change(mean_predictor, -change, log_means["predictor"], log_sds["predictor"])
  delta_substitute <- log_std_change(mean_substitution, change_sub, log_means["substitute"], log_sds["substitute"])
  
  # Draw 10,000 samples from the posterior
  prior_samples <- extract.prior(model, n = 10000)
  
  # Access the samples for the coefficients
  beta_predictor_samples <- prior_samples$b_predictor
  beta_substitute_samples <- prior_samples$b_substitute
  
  # Calculate the change in outcome for each sample
  change_in_outcome_samples <- beta_predictor_samples * delta_predictor + beta_substitute_samples * delta_substitute
  
  return(change_in_outcome_samples)
}

# Function to iterate over models and calculate change in outcome
iterate_and_calculate_results <- function(models, variable_mapping, df, change, change_fiber, output_file, trig = FALSE) {
  results <- list()
  
  # Iterate over each model
  for (model_name in names(models)) {
    model <- models[[model_name]]
    predictor_name <- variable_mapping[[model_name]]$predictor
    substitute_name <- variable_mapping[[model_name]]$substitute
    predictor <- df[[predictor_name]]
    substitute <- df[[substitute_name]]
    
    if (grepl("fiber", model_name)) {
      substitution <- change_fiber
    } else {
      substitution <- change
    }
    
    change_in_outcome_samples <- calculate_change(model, predictor, substitute, change, substitution)
    if (trig == TRUE) {
      change_in_outcome_samples <- exp(change_in_outcome_samples) - 1
    }
    
    results[[model_name]] <- change_in_outcome_samples
  }
  
  # Plotting and summarizing the results
  for (model_name in names(results)) {
    change_in_outcome_samples <- results[[model_name]]
  }
  
  # Write the results to CSV
  write.csv(data.frame(results), file = output_file, row.names = FALSE)
  
  return(results)
}

# Define quap_model function
quap_model_anthro <- function(predictor_var, substitute_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_substitute * substitute +
        b_protein * intake_protein_kcal_std +
        b_alcohol * intake_alcohol_kcal_std +
        b_fat * intake_fat_kcal_std +
        b_age * age +
        b_energy * energy,
      a_gender[gender] ~ dnorm(0, 1),
      b_predictor ~ dnorm(0, 1),
      b_age ~ dnorm(0, 1),
      b_substitute ~ dnorm(0, 1),
      b_protein ~ dnorm(0, 1),
      b_alcohol ~ dnorm(0, 1),
      b_fat ~ dnorm(0, 1),
      b_energy ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
      substitute = data[[substitute_var]],
      energy = data[[energy_var]],
      intake_protein_kcal_std = data$intake_protein_kcal_std,
      intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
      centre_id = data$centre_id,
      gender = data$gender,
      intake_fat_kcal_std = data$intake_fat_kcal_std,
      age = data$age
    ),
    control = list(maxit = 10000)
  )
}

# Define quap_model function
quap_model_blood <- function(predictor_var, substitute_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_substitute * substitute +
        b_protein * intake_protein_kcal_std +
        b_alcohol * intake_alcohol_kcal_std +
        b_fat * intake_fat_kcal_std +
        b_age * age +
        b_energy * energy +
        b_bf * bodyfat_percent_scale,
      a_gender[gender] ~ dnorm(0, 1),
      b_bf ~ dnorm(0, 1),
      b_predictor ~ dnorm(0, 1),
      b_age ~ dnorm(0, 1),
      b_substitute ~ dnorm(0, 1),
      b_protein ~ dnorm(0, 1),
      b_alcohol ~ dnorm(0, 1),
      b_fat ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      b_energy ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
      substitute = data[[substitute_var]],
      energy = data[[energy_var]],
      intake_protein_kcal_std = data$intake_protein_kcal_std,
      intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
      intake_fat_kcal_std = data$intake_fat_kcal_std,
      centre_id = data$centre_id,
      age = data$age,
      gender = data$gender,
      bodyfat_percent_scale = data$bodyfat_percent_scale
    ),
    control = list(maxit = 10000)
  )
}


# Define functions for each outcome
define_models_anthro <- function(outcome) {
  list(
    m_nmes_starch = quap_model_anthro(
      predictor_var = "intake_nmes_kcal_std",
      substitute_var = "intake_starch_kcal_std",
      energy_var = "intake_energy_kcal_starch_nmes",
      outcome_var = outcome,
      data = df
    ),
    m_starch_fiber = quap_model_anthro(
      predictor_var = "intake_starch_kcal_std",
      substitute_var = "intake_fibre_englyst_kcal_std",
      energy_var = "intake_energy_kcal_fiber_starch",
      outcome_var = outcome,
      data = df
    ),
    m_intrinsic_sugar_starch = quap_model_anthro(
      predictor_var = "intake_intrinsic_sugars_kcal_std",
      substitute_var = "intake_starch_kcal_std",
      energy_var = "intake_energy_kcal_starch_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    ),
    m_intrinsic_sugar_fiber = quap_model_anthro(
      predictor_var = "intake_intrinsic_sugars_kcal_std",
      substitute_var = "intake_fibre_englyst_kcal_std",
      energy_var = "intake_energy_kcal_fiber_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    ),
    m_nmes_fiber = quap_model_anthro(
      predictor_var = "intake_nmes_kcal_std",
      substitute_var = "intake_fibre_englyst_kcal_std",
      energy_var = "intake_energy_kcal_fiber_nmes",
      outcome_var = outcome,
      data = df
    ),
    m_nmes_intrinsic_sugar = quap_model_anthro(
      predictor_var = "intake_nmes_kcal_std",
      substitute_var = "intake_intrinsic_sugars_kcal_std",
      energy_var = "intake_energy_kcal_nmes_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    )
  )
}

# Define functions for each outcome
define_models_blood <- function(outcome) {
  list(
    m_nmes_starch = quap_model_blood(
      predictor_var = "intake_nmes_kcal_std",
      substitute_var = "intake_starch_kcal_std",
      energy_var = "intake_energy_kcal_starch_nmes",
      outcome_var = outcome,
      data = df
    ),
    m_starch_fiber = quap_model_blood(
      predictor_var = "intake_starch_kcal_std",
      substitute_var = "intake_fibre_englyst_kcal_std",
      energy_var = "intake_energy_kcal_fiber_starch",
      outcome_var = outcome,
      data = df
    ),
    m_intrinsic_sugar_starch = quap_model_blood(
      predictor_var = "intake_intrinsic_sugars_kcal_std",
      substitute_var = "intake_starch_kcal_std",
      energy_var = "intake_energy_kcal_starch_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    ),
    m_intrinsic_sugar_fiber = quap_model_blood(
      predictor_var = "intake_intrinsic_sugars_kcal_std",
      substitute_var = "intake_fibre_englyst_kcal",
      energy_var = "intake_energy_kcal_fiber_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    ),
    m_nmes_fiber = quap_model_blood(
      predictor_var = "intake_nmes_kcal_std",
      substitute_var = "intake_fibre_englyst_kcal_std",
      energy_var = "intake_energy_kcal_fiber_nmes",
      outcome_var = outcome,
      data = df
    ),
    m_nmes_intrinsic_sugar = quap_model_blood(
      predictor_var = "intake_nmes_kcal_std",
      substitute_var = "intake_intrinsic_sugars_kcal_std",
      energy_var = "intake_energy_kcal_nmes_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    )
  )
}

# Define predictors and substitutions for each model based on the model names
variable_mapping <- list(
  m_nmes_starch = list(predictor = "intake_nmes_kcal", substitute = "intake_starch_kcal"),
  m_starch_fiber = list(predictor = "intake_starch_kcal", substitute = "intake_fibre_englyst_kcal"),
  m_intrinsic_sugar_starch = list(predictor = "intake_intrinsic_sugars_kcal", substitute = "intake_starch_kcal"),
  m_intrinsic_sugar_fiber = list(predictor = "intake_intrinsic_sugars_kcal", substitute = "intake_fibre_englyst_kcal"),
  m_nmes_fiber = list(predictor = "intake_nmes_kcal", substitute = "intake_fibre_englyst_kcal"),
  m_nmes_intrinsic_sugar = list(predictor = "intake_nmes_kcal", substitute = "intake_intrinsic_sugars_kcal")
)

# Store models for bodyfat_percent in a list
models_bf <- define_models_anthro("bodyfat_percent")

# Specify output file path
output_file <- "../results/substitutions_prior/bodyfat_substitution_results.csv"

# Call the function to iterate over models, calculate results, and generate outputs
iterate_and_calculate_results(models_bf, variable_mapping, df, change, change_fiber, output_file)

df$trig_log <- log(df$trig)
models_trig <- define_models_blood("trig_log")

# Specify output file path
output_file <- "../results/substitutions_prior/trig_substitution_results.csv"

# Call the function to iterate over models, calculate results, and generate outputs
iterate_and_calculate_results(models_trig, variable_mapping, df, change, change_fiber, output_file, trig = TRUE)

models_hdl <- define_models_blood("hdl")

# Specify output file path
output_file <- "../results/substitutions_prior/hdl_substitution_results.csv"

# Call the function to iterate over models, calculate results, and generate outputs
iterate_and_calculate_results(models_hdl, variable_mapping, df, change, change_fiber, output_file)

models_ldl <- define_models_blood("ldl")

# Specify output file path
output_file <- "../results/substitutions_prior/ldl_substitution_results.csv"

# Call the function to iterate over models, calculate results, and generate outputs
iterate_and_calculate_results(models_ldl, variable_mapping, df, change, change_fiber, output_file)

models_hba1c <- define_models_blood("hba1c_percent")

# Specify output file path
output_file <- "../results/substitutions_prior/hba1c_substitution_results.csv"

# Call the function to iterate over models, calculate results, and generate outputs
iterate_and_calculate_results(models_hba1c, variable_mapping, df, change, change_fiber, output_file)

models_bmi <- define_models_anthro("bmi")

# Specify output file path
output_file <- "../results/substitutions_prior/bmi_substitution_results.csv"

# Call the function to iterate over models, calculate results, and generate outputs
iterate_and_calculate_results(models_bmi, variable_mapping, df, change, change_fiber, output_file)

models_wc <- define_models_anthro("waistcirumference")

# Specify output file path
output_file <- "../results/substitutions_prior/waistcirumference_substitution_results.csv"


# Call the function to iterate over models, calculate results, and generate outputs
iterate_and_calculate_results(models_wc, variable_mapping, df, change, change_fiber, output_file)
