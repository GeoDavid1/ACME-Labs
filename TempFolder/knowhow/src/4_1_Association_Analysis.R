# Load necessary libraries
library(rethinking)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the dataset
df_anthro <- read.csv("../data/processed_files/subsetted_df_anthro.csv")
df_blood <- read.csv("../data/processed_files/subsetted_df_blood.csv")

define_models_anthro_association <- function(outcome, df) {
  list(
    m_nmes = quap_model_anthro_assoc(
      predictor_var = "intake_nmes_kcal_std",
      energy_var = "intake_energy_kcal_std_nmes",
      outcome_var = outcome,
      data = df
    ),
    m_fiber = quap_model_anthro_assoc(
      predictor_var = "intake_fibre_englyst_kcal_std",
      energy_var = "intake_energy_kcal_std_fiber",
      outcome_var = outcome,
      data = df
    ),
    m_starch = quap_model_anthro_assoc(
      predictor_var = "intake_starch_kcal_std",
      energy_var = "intake_energy_kcal_std_starch",
      outcome_var = outcome,
      data = df
    ),
    m_intrinsic_sugars = quap_model_anthro_assoc(
      predictor_var = "intake_intrinsic_sugars_kcal_std",
      energy_var = "intake_energy_kcal_std_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    ),
    m_sucrose = quap_model_anthro_assoc(
      predictor_var = "intake_sucrose_kcal_std",
      energy_var = "intake_energy_kcal_std_sucrose",
      outcome_var = outcome,
      data = df
    ),
    m_total_sugars = quap_model_anthro_assoc(
      predictor_var = "intake_total_sugars_kcal_std",
      energy_var = "intake_energy_kcal_std_total_sugars",
      outcome_var = outcome,
      data = df
    )
  )
}

define_models_blood_association <- function(outcome, df) {
  list(
    m_nmes = quap_model_blood_assoc(
      predictor_var = "intake_nmes_kcal_std",
      energy_var = "intake_energy_kcal_std_nmes",
      outcome_var = outcome,
      data = df
    ),
    m_fiber = quap_model_blood_assoc(
      predictor_var = "intake_fibre_englyst_kcal_std",
      energy_var = "intake_energy_kcal_std_fiber",
      outcome_var = outcome,
      data = df
    ),
    m_starch = quap_model_blood_assoc(
      predictor_var = "intake_starch_kcal_std",
      energy_var = "intake_energy_kcal_std_starch",
      outcome_var = outcome,
      data = df
    ),
    m_intrinsic_sugars = quap_model_blood_assoc(
      predictor_var = "intake_intrinsic_sugars_kcal_std",
      energy_var = "intake_energy_kcal_std_intrinsic_sugar",
      outcome_var = outcome,
      data = df
    ),
    m_sucrose = quap_model_blood_assoc(
      predictor_var = "intake_sucrose_kcal_std",
      energy_var = "intake_energy_kcal_std_sucrose",
      outcome_var = outcome,
      data = df
    ),
    m_total_sugars = quap_model_blood_assoc(
      predictor_var = "intake_total_sugars_kcal_std",
      energy_var = "intake_energy_kcal_std_total_sugars",
      outcome_var = outcome,
      data = df
    )
  )
}

# Function to calculate posterior predictions for a specific predictor value
predict_posterior <- function(predictor_value, model, covariates) {
  covariates$predictor <- predictor_value
  link(model, data = covariates)
}

# Function to process outcome and generate results
process_outcome <- function(outcome, df, anthro = TRUE, trig = FALSE) {
  if (anthro) {
    model_list <- define_models_anthro_association(outcome, df)
  } else {
    model_list <- define_models_blood_association(outcome, df)
  }
  
  results <- list()
  
  for (model_name in names(model_list)) {
    fitted_model <- model_list[[model_name]]
    
    # Generate posterior samples
    posterior_samples <- extract.samples(fitted_model)
    
    # Define covariates to their mean or specific values
    covariate_means <- data.frame(
      intake_protein_kcal_std = mean(df$intake_protein_kcal_std),
      intake_alcohol_kcal_std = mean(df$intake_alcohol_kcal_std),
      intake_fat_kcal_std = mean(df$intake_fat_kcal_std),
      age = mean(df$age),
      intake_fiber_kcal_std = mean(df$intake_fiber_kcal_std),
      gender = 2,
      bodyfat_percent_scale = mean(df$bodyfat_percent_scale),
      energy = mean(df[[paste0("intake_energy_kcal_std_", sub("m_", "", model_name))]]),
      centre_id = 1  # Assuming using the first center as reference
    )
    
    # Calculate the posterior predictions for each tertile
    intake_tertiles <- quantile(df[[paste0("intake_", sub("m_", "", model_name), "_kcal_std")]], probs = c(0.33, 0.66))
    
    # Calculate for each tertile
    #beware, models with fiber twice in them need to pass the correct intake in for accurate prediction!
    if (sub("m_", "", model_name) == 'fiber'){
      covariate_means$intake_fiber_kcal_std <- intake_tertiles[1]
    }
    low_tert <- predict_posterior(intake_tertiles[1], fitted_model, covariate_means)
    if (sub("m_", "", model_name) == 'fiber'){
      covariate_means$intake_fiber_kcal_std <- intake_tertiles[2]
    }
    mid_tert <- predict_posterior(intake_tertiles[2], fitted_model, covariate_means)
    if (sub("m_", "", model_name) == 'fiber'){
      covariate_means$intake_fiber_kcal_std <- max(df[[paste0("intake_", sub("m_", "", model_name), "_kcal_std")]])
    }
    high_tert <- predict_posterior(max(df[[paste0("intake_", sub("m_", "", model_name), "_kcal_std")]]), fitted_model, covariate_means)
    
    # Summarize the results
    low_tert_mean <- mean(low_tert)
    low_tert_ci <- PI(low_tert, prob = 0.95)
    
    mid_tert_mean <- mean(mid_tert)
    mid_tert_ci <- PI(mid_tert, prob = 0.95)
    
    high_tert_mean <- mean(high_tert)
    high_tert_ci <- PI(high_tert, prob = 0.95)
    
    # trig is using a log outcome, so exponentiate the predictions
    
    if (trig == TRUE) {
      
      # Here, I should be able to calculate the contrast between low and high tertiles. 
      contrast = exp(low_tert) - exp(high_tert)
      # Compute the probability that the difference is greater than zero
      contrast_mean = mean(contrast > 0)
      # Compute credible intervals
      contrast_ci = PI(contrast, prob = 0.95)

      # Create a summary table
      result_table <- data.frame(
        Outcome = outcome,
        Predictor = model_name,
        Tertiles = c("Low (T1)", "Moderate (T2)", "High (T3)"),
        Mean = c(exp(low_tert_mean), exp(mid_tert_mean), exp(high_tert_mean)),
        CI_lower = c(exp(low_tert_ci[1]), exp(mid_tert_ci[1]), exp(high_tert_ci[1])),
        CI_upper = c(exp(low_tert_ci[2]), exp(mid_tert_ci[2]), exp(high_tert_ci[2])),
        Contrast = contrast_mean,
        Contrast_CI_lower = contrast_ci[1],
        Contrast_CI_upper = contrast_ci[2]
      )
    } else {
      
      # Here, I should be able to calculate the contrast between low and high tertiles. 
      contrast = low_tert - high_tert
      
      # Compute the probability that the difference is greater than zero
      contrast_mean = mean(contrast > 0)
      # Compute credible intervals
      contrast_ci = PI(contrast, prob = 0.95)
      
      # Create a summary table
      result_table <- data.frame(
        Outcome = outcome,
        Predictor = model_name,
        Tertiles = c("Low (T1)", "Moderate (T2)", "High (T3)"),
        Mean = c(low_tert_mean, mid_tert_mean, high_tert_mean),
        CI_lower = c(low_tert_ci[1], mid_tert_ci[1], high_tert_ci[1]),
        CI_upper = c(low_tert_ci[2], mid_tert_ci[2], high_tert_ci[2]),
        Contrast = contrast_mean,
        Contrast_CI_lower = contrast_ci[1],
        Contrast_CI_upper = contrast_ci[2]
      )
    }
    results[[model_name]] <- result_table
  }
  
  return(do.call(rbind, results))
}

# Define the outcome variable
anthro_outcomes <- c("waistcirumference", "bmi", "bodyfat_percent")
blood_outcomes <- c("ldl", "hdl", "hba1c_percent")

# We need to repeat this for:
# Model 1: Age, sex, centre_id
# Model 2: + fiber, protein, sat.fat, alcohol
# Model 3: + remaining energy

# Model 1
quap_model_anthro_assoc <- function(predictor_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_age * age,
      a_gender[gender] ~ dnorm(0, 1),
      b_predictor ~ dnorm(0, 1),
      b_age ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
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

quap_model_blood_assoc <- function(predictor_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_age * age,
      a_gender[gender] ~ dnorm(0, 1),
      b_predictor ~ dnorm(0, 1),
      b_age ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
      energy = data[[energy_var]],
      intake_protein_kcal_std = data$intake_protein_kcal_std,
      intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
      intake_fat_kcal_std = data$intake_fat_kcal_std,
      centre_id = data$centre_id,
      gender = data$gender,
      age = data$age,
      bodyfat_percent_scale = data$bodyfat_percent_scale
    ),
    control = list(maxit = 10000)
  )
}

# Process all outcomes
all_results <- list()

for (outcome in anthro_outcomes) {
  all_results[[outcome]] <- process_outcome(outcome, df_anthro, anthro = TRUE)
}

for (outcome in blood_outcomes) {
  all_results[[outcome]] <- process_outcome(outcome, df_blood, anthro = FALSE)
}

all_results[["trig"]] <- process_outcome("trig_log", df_blood, anthro = FALSE, trig = TRUE)

all_results_df <- do.call(rbind, all_results)
write.csv(all_results_df, file = "../results/associations/association_table_model_1.csv", row.names = FALSE)


# Model 2
quap_model_anthro_assoc <- function(predictor_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_protein * intake_protein_kcal_std +
        b_alcohol * intake_alcohol_kcal_std +
        b_fat * intake_fat_kcal_std +
        b_fiber * intake_fiber_kcal_std + 
        b_age * age,
      a_gender[gender] ~ dnorm(0, 1),
      b_predictor ~ dnorm(0, 1),
      b_age ~ dnorm(0, 1),
      b_protein ~ dnorm(0, 1),
      b_fiber ~ dnorm(0, 1),
      b_alcohol ~ dnorm(0, 1),
      b_fat ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
      energy = data[[energy_var]],
      intake_protein_kcal_std = data$intake_protein_kcal_std,
      intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
      centre_id = data$centre_id,
      intake_fiber_kcal_std = data$intake_fiber_kcal_std,
      gender = data$gender,
      intake_fat_kcal_std = data$intake_fat_kcal_std,
      age = data$age
    ),
    control = list(maxit = 10000)
  )
}

quap_model_blood_assoc <- function(predictor_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_protein * intake_protein_kcal_std +
        b_alcohol * intake_alcohol_kcal_std +
        b_fat * intake_fat_kcal_std +
        b_fiber * intake_fiber_kcal_std + 
        b_age * age +
        b_bf * bodyfat_percent_scale,
      a_gender[gender] ~ dnorm(0, 1),
      b_bf ~ dnorm(0, 1),
      b_predictor ~ dnorm(0, 1),
      b_age ~ dnorm(0, 1),
      b_fiber ~ dnorm(0, 1),
      b_protein ~ dnorm(0, 1),
      b_alcohol ~ dnorm(0, 1),
      b_fat ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
      energy = data[[energy_var]],
      intake_protein_kcal_std = data$intake_protein_kcal_std,
      intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
      intake_fat_kcal_std = data$intake_fat_kcal_std,
      intake_fiber_kcal_std = data$intake_fiber_kcal_std,
      centre_id = data$centre_id,
      gender = data$gender,
      age = data$age,
      bodyfat_percent_scale = data$bodyfat_percent_scale
    ),
    control = list(maxit = 10000)
  )
}

# Process all outcomes
all_results <- list()

for (outcome in anthro_outcomes) {
  all_results[[outcome]] <- process_outcome(outcome, df_anthro, anthro = TRUE)
}

for (outcome in blood_outcomes) {
  all_results[[outcome]] <- process_outcome(outcome, df_blood, anthro = FALSE)
}

all_results[["trig"]] <- process_outcome("trig_log", df_blood, anthro = FALSE, trig = TRUE)

all_results_df <- do.call(rbind, all_results)
write.csv(all_results_df, file = "../results/associations/association_table_model_2.csv", row.names = FALSE)

# Model 3
quap_model_anthro_assoc <- function(predictor_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_protein * intake_protein_kcal_std +
        b_alcohol * intake_alcohol_kcal_std +
        b_fiber * intake_fiber_kcal_std + 
        b_fat * intake_fat_kcal_std +
        b_age * age +
        b_energy * energy,
      a_gender[gender] ~ dnorm(0, 1),
      b_predictor ~ dnorm(0, 1),
      b_age ~ dnorm(0, 1),
      b_protein ~ dnorm(0, 1),
      b_alcohol ~ dnorm(0, 1),
      b_fat ~ dnorm(0, 1),
      b_fiber ~ dnorm(0, 1),
      b_energy ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
      energy = data[[energy_var]],
      intake_protein_kcal_std = data$intake_protein_kcal_std,
      intake_fiber_kcal_std = data$intake_fiber_kcal_std,
      intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
      centre_id = data$centre_id,
      gender = data$gender,
      intake_fat_kcal_std = data$intake_fat_kcal_std,
      age = data$age
    ),
    control = list(maxit = 10000)
  )
}

quap_model_blood_assoc <- function(predictor_var, energy_var, outcome_var, data) {
  quap(
    alist(
      outcome ~ dnorm(mu, sigma),
      mu <- a_gender[gender] + 
        b_center[centre_id] +
        b_predictor * predictor +
        b_fiber * intake_fiber_kcal_std + 
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
      b_protein ~ dnorm(0, 1),
      b_alcohol ~ dnorm(0, 1),
      b_fiber ~ dnorm(0, 1),
      b_fat ~ dnorm(0, 1),
      b_center[centre_id] ~ dnorm(0, 1),
      b_energy ~ dnorm(0, 1),
      sigma ~ dexp(1)
    ),
    data = data.frame(
      outcome = data[[outcome_var]],
      predictor = data[[predictor_var]],
      energy = data[[energy_var]],
      intake_protein_kcal_std = data$intake_protein_kcal_std,
      intake_fiber_kcal_std = data$intake_fiber_kcal_std,
      intake_alcohol_kcal_std = data$intake_alcohol_kcal_std,
      intake_fat_kcal_std = data$intake_fat_kcal_std,
      centre_id = data$centre_id,
      gender = data$gender,
      age = data$age,
      bodyfat_percent_scale = data$bodyfat_percent_scale
    ),
    control = list(maxit = 10000)
  )
}

# Process all outcomes
all_results <- list()

for (outcome in anthro_outcomes) {
  all_results[[outcome]] <- process_outcome(outcome, df_anthro, anthro = TRUE)
}

for (outcome in blood_outcomes) {
  all_results[[outcome]] <- process_outcome(outcome, df_blood, anthro = FALSE)
}

all_results[["trig"]] <- process_outcome("trig_log", df_blood, anthro = FALSE, trig = TRUE)


all_results_df <- do.call(rbind, all_results)
write.csv(all_results_df, file = "../results/associations/association_table_model_3.csv", row.names = FALSE)

#need to get the tertiles of intake 

library(boot)

# Function to compute quantiles
quantile_func <- function(data, indices, probs) {
  return(quantile(data[indices], probs = probs))
}

# Function to format results with CI
format_result <- function(boot_result, probs) {
  result <- sapply(1:length(probs), function(i) {
    estimate <- round(boot_result$t0[i], 2)
    ci <- round(boot.ci(boot_result, type = "perc", index = i)$percent[4:5], 2)
    sprintf("%.2f\n(%.2f, %.2f)", estimate, ci[1], ci[2])
  })
  names(result) <- paste0(c("33", "66", "99"), "%")
  return(result)
}

# Set random seed for reproducibility
set.seed(123)

# Compute bootstrap for each variable
probs <- c(0.33, 0.66, 0.99)
n_boot <- 1000

df <- df_anthro

nmes_boot <- boot(df$intake_nmes, quantile_func, R = n_boot, probs = probs)
intrinsic_sugar_boot <- boot(df$intake_intrinsic_sugars, quantile_func, R = n_boot, probs = probs)
starch_boot <- boot(df$intake_starch, quantile_func, R = n_boot, probs = probs)
fibre_boot <- boot(df$intake_fibre_englyst, quantile_func, R = n_boot, probs = probs)

# Format results
result <- data.frame(
  NMES_33 = format_result(nmes_boot, probs)[1],
  NMES_66 = format_result(nmes_boot, probs)[2],
  NMES_99 = format_result(nmes_boot, probs)[3],
  Intrinsic_Sugar_33 = format_result(intrinsic_sugar_boot, probs)[1],
  Intrinsic_Sugar_66 = format_result(intrinsic_sugar_boot, probs)[2],
  Intrinsic_Sugar_99 = format_result(intrinsic_sugar_boot, probs)[3],
  Starch_33 = format_result(starch_boot, probs)[1],
  Starch_66 = format_result(starch_boot, probs)[2],
  Starch_99 = format_result(starch_boot, probs)[3],
  Fibre_33 = format_result(fibre_boot, probs)[1],
  Fibre_66 = format_result(fibre_boot, probs)[2],
  Fibre_99 = format_result(fibre_boot, probs)[3]
)

write.csv(result, "../results/associations/tertiles_of_intakes_output_anthro.csv", row.names = FALSE)

df <- df_blood

nmes_boot <- boot(df$intake_nmes, quantile_func, R = n_boot, probs = probs)
intrinsic_sugar_boot <- boot(df$intake_intrinsic_sugars, quantile_func, R = n_boot, probs = probs)
starch_boot <- boot(df$intake_starch, quantile_func, R = n_boot, probs = probs)
fibre_boot <- boot(df$intake_fibre_englyst, quantile_func, R = n_boot, probs = probs)

# Format results
result <- data.frame(
  NMES_33 = format_result(nmes_boot, probs)[1],
  NMES_66 = format_result(nmes_boot, probs)[2],
  NMES_99 = format_result(nmes_boot, probs)[3],
  Intrinsic_Sugar_33 = format_result(intrinsic_sugar_boot, probs)[1],
  Intrinsic_Sugar_66 = format_result(intrinsic_sugar_boot, probs)[2],
  Intrinsic_Sugar_99 = format_result(intrinsic_sugar_boot, probs)[3],
  Starch_33 = format_result(starch_boot, probs)[1],
  Starch_66 = format_result(starch_boot, probs)[2],
  Starch_99 = format_result(starch_boot, probs)[3],
  Fibre_33 = format_result(fibre_boot, probs)[1],
  Fibre_66 = format_result(fibre_boot, probs)[2],
  Fibre_99 = format_result(fibre_boot, probs)[3]
)

write.csv(result, "../results/associations/tertiles_of_intakes_output_blood.csv", row.names = FALSE)