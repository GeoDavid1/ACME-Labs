# Load necessary libraries
library(rethinking)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the dataset
df_list <- list(
  blood = read_csv("../data/processed_files/subsetted_df_blood.csv"),
  anthro = read_csv("../data/processed_files/subsetted_df_anthro.csv")
)

# Loop over the dataframes
for (df_name in names(df_list)) {
  df <- df_list[[df_name]]
  # Compute summary statistics
  summary_stats <- df %>%
    summarize(
      age_mean = mean(age, na.rm = TRUE),
      age_std = sd(age, na.rm = TRUE),
      male_percent = mean(gender == 1, na.rm = TRUE) * 100,
      energy_mean_mj = mean(intake_energy_mj, na.rm = TRUE) * 239.005736,
      energy_std_mj = sd(intake_energy_mj, na.rm = TRUE) * 239.005736,
      dietary_fat_mean = mean(intake_fat, na.rm = TRUE),
      dietary_fat_std = sd(intake_fat, na.rm = TRUE),
      fibre_mean = mean(intake_fibre_englyst, na.rm = TRUE),
      fibre_std = sd(intake_fibre_englyst, na.rm = TRUE),
      protein_mean = mean(intake_protein, na.rm = TRUE),
      protein_std = sd(intake_protein, na.rm = TRUE),
      carbohydrate_mean = mean(intake_carbohydrate, na.rm = TRUE),
      carbohydrate_std = sd(intake_carbohydrate, na.rm = TRUE),
      fructose_mean = mean(intake_fructose, na.rm = TRUE),
      fructose_std = sd(intake_fructose, na.rm = TRUE),
      glucose_mean = mean(intake_glucose, na.rm = TRUE),
      glucose_std = sd(intake_glucose, na.rm = TRUE),
      intrinsic_sugars_mean = mean(intake_intrinsic_sugars, na.rm = TRUE),
      intrinsic_sugars_std = sd(intake_intrinsic_sugars, na.rm = TRUE),
      nmes_mean = mean(intake_nmes, na.rm = TRUE),
      nmes_std = sd(intake_nmes, na.rm = TRUE),
      starch_mean = mean(intake_starch, na.rm = TRUE),
      starch_std = sd(intake_starch, na.rm = TRUE),
      sucrose_mean = mean(intake_sucrose, na.rm = TRUE),
      sucrose_std = sd(intake_sucrose, na.rm = TRUE),
      total_sugars_mean = mean(intake_total_sugars, na.rm = TRUE),
      total_sugars_std = sd(intake_total_sugars, na.rm = TRUE),
      alcohol_mean = mean(intake_alcohol, na.rm = TRUE),
      alcohol_std = sd(intake_alcohol, na.rm = TRUE),
      cholesterol_mean = mean(intake_cholesterol, na.rm = TRUE),
      cholesterol_std = sd(intake_cholesterol, na.rm = TRUE),
      saturated_fat_mean = mean(intake_satd_fa, na.rm = TRUE),
      saturated_fat_std = sd(intake_satd_fa, na.rm = TRUE),
      bmi_mean = mean(bmi, na.rm = TRUE),
      bmi_std = sd(bmi, na.rm = TRUE),
      bodyfat_kg_mean = mean(bodyfat, na.rm = TRUE),
      bodyfat_kg_std = sd(bodyfat, na.rm = TRUE),
      bodyfat_percent_mean = mean(bodyfat_percent, na.rm = TRUE),
      bodyfat_percent_std = sd(bodyfat_percent, na.rm = TRUE),
      waist_circumference_cm_mean = mean(waistcirumference, na.rm = TRUE),
      waist_circumference_cm_std = sd(waistcirumference, na.rm = TRUE),
      hba1c_mmol_mol_mean = mean(hba1c, na.rm = TRUE),
      hba1c_mmol_mol_std = sd(hba1c, na.rm = TRUE),
      hba1c_percent_mean = mean(hba1c_percent, na.rm = TRUE),
      hba1c_percent_std = sd(hba1c_percent, na.rm = TRUE),
      hdl_mean = mean(hdl, na.rm = TRUE),
      hdl_std = sd(hdl, na.rm = TRUE),
      ldl_mean = mean(ldl, na.rm = TRUE),
      ldl_std = sd(ldl, na.rm = TRUE),
      triglycerides_mean = mean(trig, na.rm = TRUE),
      triglycerides_std = sd(trig, na.rm = TRUE)
    )
  
  # Calculate the number of non-NA entries for each variable
  n_entries <- colSums(!is.na(df))
  
  # Create a new dataframe for the output
  output <- data.frame(
    Variable = c("Age (years)", "% Male", 
                 "Energy (MJ)", "Dietary Fat (g)", "Saturated Fat (g)", "Protein (g)", 
                 "Carbohydrate (g)", "Fibre Englyst (g)", "Fructose (g)", "Glucose (g)", "Intrinsic Sugars (g)", 
                 "NMES (g)", "Starch (g)", "Sucrose (g)", "Total Sugars (g)", 
                 "Alcohol (g)", "Cholesterol (mg)", 
                 "BMI (kg/m2)", "Bodyfat (kg)", "Bodyfat Percent (%)", 
                 "Waist Circumference (cm)",  "Hba1c Percent (%)", 
                 "HDL (mmol/L)", "LDL (mmol/L)", "Triglycerides (mmol/L)"),
    N = c(n_entries[["age"]], n_entries[["gender"]],
          n_entries[["intake_energy_mj"]], n_entries[["intake_fat"]], n_entries[["intake_satd_fa"]], n_entries[["intake_protein"]],
          n_entries[["intake_carbohydrate"]], n_entries[["intake_fibre_englyst"]], n_entries[["intake_fructose"]], n_entries[["intake_glucose"]], n_entries[["intake_intrinsic_sugars"]],
          n_entries[["intake_nmes"]], n_entries[["intake_starch"]], n_entries[["intake_sucrose"]], n_entries[["intake_total_sugars"]],
          n_entries[["intake_alcohol"]], n_entries[["intake_cholesterol"]],
          n_entries[["bmi"]], n_entries[["bodyfat"]], n_entries[["bodyfat_percent"]],
          n_entries[["waistcirumference"]],  n_entries[["hba1c_percent"]],
          n_entries[["hdl"]], n_entries[["ldl"]], n_entries[["trig"]]),
    Mean_Std = c(
      sprintf("%.2f (%.2f)", summary_stats$age_mean, summary_stats$age_std),
      sprintf("%.2f", summary_stats$male_percent),
      sprintf("%.2f (%.2f)", summary_stats$energy_mean_mj, summary_stats$energy_std_mj),
      sprintf("%.2f (%.2f)", summary_stats$dietary_fat_mean, summary_stats$dietary_fat_std),
      sprintf("%.2f (%.2f)", summary_stats$saturated_fat_mean, summary_stats$saturated_fat_std),
      sprintf("%.2f (%.2f)", summary_stats$protein_mean, summary_stats$protein_std),
      sprintf("%.2f (%.2f)", summary_stats$carbohydrate_mean, summary_stats$carbohydrate_std),
      sprintf("%.2f (%.2f)", summary_stats$fibre_mean, summary_stats$fibre_std),
      sprintf("%.2f (%.2f)", summary_stats$fructose_mean, summary_stats$fructose_std),
      sprintf("%.2f (%.2f)", summary_stats$glucose_mean, summary_stats$glucose_std),
      sprintf("%.2f (%.2f)", summary_stats$intrinsic_sugars_mean, summary_stats$intrinsic_sugars_std),
      sprintf("%.2f (%.2f)", summary_stats$nmes_mean, summary_stats$nmes_std),
      sprintf("%.2f (%.2f)", summary_stats$starch_mean, summary_stats$starch_std),
      sprintf("%.2f (%.2f)", summary_stats$sucrose_mean, summary_stats$sucrose_std),
      sprintf("%.2f (%.2f)", summary_stats$total_sugars_mean, summary_stats$total_sugars_std),
      sprintf("%.2f (%.2f)", summary_stats$alcohol_mean, summary_stats$alcohol_std),
      sprintf("%.2f (%.2f)", summary_stats$cholesterol_mean, summary_stats$cholesterol_std),
      sprintf("%.2f (%.2f)", summary_stats$bmi_mean, summary_stats$bmi_std),
      sprintf("%.2f (%.2f)", summary_stats$bodyfat_kg_mean, summary_stats$bodyfat_kg_std),
      sprintf("%.2f (%.2f)", summary_stats$bodyfat_percent_mean, summary_stats$bodyfat_percent_std),
      sprintf("%.2f (%.2f)", summary_stats$waist_circumference_cm_mean, summary_stats$waist_circumference_cm_std),
      sprintf("%.2f (%.2f)", summary_stats$hba1c_percent_mean, summary_stats$hba1c_percent_std),
      sprintf("%.2f (%.2f)", summary_stats$hdl_mean, summary_stats$hdl_std),
      sprintf("%.2f (%.2f)", summary_stats$ldl_mean, summary_stats$ldl_std),
      sprintf("%.2f (%.2f)", summary_stats$triglycerides_mean, summary_stats$triglycerides_std)
    )
  )
  
  # Save to CSV
  write.csv(output, paste0("../results/characteristics_tables/characteristics_table_", df_name, ".csv"), row.names = FALSE)
}
