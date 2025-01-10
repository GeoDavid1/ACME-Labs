# Load necessary libraries
library(readxl)
library(rethinking)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Define a list of dataframes to process
df_list <- list(
  blood = read_excel("../data/processed_files/aggregrated_data_blood.xlsx"),
  anthro = read_excel("../data/processed_files/aggregrated_data_anthro.xlsx")
)

# Loop over the dataframes
for (df_name in names(df_list)) {
  print(df_name)
  df_raw <- df_list[[df_name]]
  # Select the original columns
  original_columns <- c("intake_protein", 
                        "intake_sucrose",
                        "intake_total_sugars",
                        "intake_fat",                                
                        "intake_carbohydrate",                       
                        "intake_energy_kj",                          
                        "intake_alcohol",                            
                        "intake_fibre_englyst",                     
                        "intake_nmes",                                
                        "intake_intrinsic_sugars",                   
                        "intake_starch",                             
                        "intake_energy_mj",                          
                        "centre_id",                                 
                        "ldl",                                    
                        "hdl",                                     
                        "trig",                                     
                        "hba1c_percent",                              
                        "bodyfat_percent",                           
                        "age",  
                        "intake_satd_fa",
                        "gender",                                    
                        "waistcirumference",                         
                        "bmi")
  
  # Select the additional columns to include
  additional_columns <- c(
    "bodyfat", 
    "hba1c",
    "intake_fructose",
    "intake_glucose",
    "intake_cholesterol")
  
  # Subset the data to include both original and additional columns
  df <- subset(df_raw, select = c(original_columns, additional_columns))
  
  # Perform complete cases operation on the original columns only
  df <- df[complete.cases(df[original_columns]), ]
  
  
  # Drop rows with zero values except 'intake_alcohol'
  cols <- setdiff(names(df), "intake_alcohol")
  df <- df[!apply(df[cols] == 0, 1, any), ]
  
  
  # Drop unrealistic intake levels
  df <- subset(df, intake_energy_kj >= 600/0.239006 & intake_energy_kj <= 4200/0.239006)
  
  #Get variables on same units:
  df$intake_protein_kcal <- df$intake_protein * 4 
  df$intake_sucrose_kcal <- df$intake_sucrose * 4 
  df$intake_total_sugars_kcal <- df$intake_total_sugars * 4 
  df$intake_fat_kcal <- df$intake_satd_fa * 9 #beware, shortcut!
  df$intake_alcohol_kcal <- df$intake_alcohol * 7 + 1
  df$intake_fibre_englyst_kcal <- df$intake_fibre_englyst * 2 
  df$intake_nmes_kcal <- df$intake_nmes * 4 
  df$intake_intrinsic_sugars_kcal <- df$intake_intrinsic_sugars * 4 
  df$intake_starch_kcal <- df$intake_starch * 4 
  df$intake_energy_kcal <- df$intake_energy_kj * 0.239006 - df$intake_protein_kcal - df$intake_fat_kcal - df$intake_alcohol_kcal - df$intake_fibre_englyst_kcal
  
  # Calculate remaining energies
  df$intake_energy_kcal_fiber <- df$intake_energy_kcal 
  df$intake_energy_kcal_starch <- df$intake_energy_kcal - df$intake_starch_kcal
  df$intake_energy_kcal_nmes <- df$intake_energy_kcal - df$intake_nmes_kcal
  df$intake_energy_kcal_intrinsic_sugar <- df$intake_energy_kcal - df$intake_intrinsic_sugars_kcal
  df$intake_energy_kcal_total_sugars <- df$intake_energy_kcal - df$intake_total_sugars_kcal
  df$intake_energy_kcal_sucrose <- df$intake_energy_kcal - df$intake_sucrose_kcal
  
  df$intake_energy_kcal_fiber_starch <- df$intake_energy_kcal - df$intake_starch_kcal 
  df$intake_energy_kcal_fiber_nmes <- df$intake_energy_kcal - df$intake_nmes_kcal 
  df$intake_energy_kcal_fiber_intrinsic_sugar <- df$intake_energy_kcal - df$intake_intrinsic_sugars_kcal 
  
  df$intake_energy_kcal_starch_nmes <- df$intake_energy_kcal - df$intake_nmes_kcal - df$intake_starch_kcal
  df$intake_energy_kcal_starch_intrinsic_sugar <- df$intake_energy_kcal - df$intake_intrinsic_sugars_kcal - df$intake_starch_kcal
  
  df$intake_energy_kcal_nmes_intrinsic_sugar <- df$intake_energy_kcal - df$intake_intrinsic_sugars_kcal - df$intake_nmes_kcal
  
  #additional energy variables for leave-one-out sensitivity analysis
  df$intake_energy_kcal_nmes_protein <- df$intake_energy_kcal_nmes + df$intake_protein_kcal
  df$intake_energy_kcal_nmes_fat <- df$intake_energy_kcal_nmes + df$intake_fat_kcal
  df$intake_energy_kcal_nmes_alcohol <- df$intake_energy_kcal_nmes + df$intake_alcohol_kcal
  
  df$intake_energy_kcal_intrinsic_sugar_protein <- df$intake_energy_kcal_intrinsic_sugar + df$intake_protein_kcal
  df$intake_energy_kcal_intrinsic_sugar_fat <- df$intake_energy_kcal_intrinsic_sugar + df$intake_fat_kcal
  df$intake_energy_kcal_intrinsic_sugar_alcohol <- df$intake_energy_kcal_intrinsic_sugar + df$intake_alcohol_kcal
  
  
  # Filter out non-realistic intake values
  df <- subset(df, intake_protein_kcal >= 0
               & intake_energy_kcal_fiber >= 0
               & intake_energy_kcal_starch >= 0
               & intake_energy_kcal_nmes >= 0
               & intake_energy_kcal_sucrose >= 0
               & intake_energy_kcal_total_sugars >= 0
               & intake_energy_kcal_intrinsic_sugar >= 0
               & intake_energy_kcal_fiber_starch >= 0
               & intake_energy_kcal_fiber_nmes >= 0
               & intake_energy_kcal_fiber_intrinsic_sugar >= 0
               & intake_energy_kcal_starch_nmes >= 0
               & intake_energy_kcal_starch_intrinsic_sugar >= 0
               & intake_energy_kcal_nmes_intrinsic_sugar >= 0)
  
  # Log normalize and then standardize all variables
  selected_cols <- df[, 
                      c(
                        "intake_protein_kcal", 
                        "intake_fat_kcal", 
                        "intake_alcohol_kcal",
                        "intake_fibre_englyst_kcal",
                        "intake_nmes_kcal",
                        "intake_total_sugars_kcal",
                        "intake_sucrose_kcal",
                        "intake_intrinsic_sugars_kcal", 
                        "intake_starch_kcal",
                        "intake_energy_kcal",
                        "ldl"    ,                                    
                        "hdl"   ,                                     
                        "trig"  ,                                     
                        "hba1c_percent",                              
                        "bodyfat_percent" ,                           
                        "waistcirumference" ,                         
                        "bmi", 
                        "intake_energy_kcal_fiber", 
                        'intake_energy_kcal_starch',
                        'intake_energy_kcal_nmes',
                        'intake_energy_kcal_total_sugars',
                        'intake_energy_kcal_sucrose',
                        'intake_energy_kcal_intrinsic_sugar',
                        'intake_energy_kcal_fiber_starch',
                        'intake_energy_kcal_fiber_nmes',
                        'intake_energy_kcal_fiber_intrinsic_sugar',
                        'intake_energy_kcal_starch_nmes',
                        'intake_energy_kcal_starch_intrinsic_sugar',
                        'intake_energy_kcal_nmes_intrinsic_sugar',
                        'intake_energy_kcal_nmes_protein',
                        'intake_energy_kcal_nmes_fat',
                        'intake_energy_kcal_nmes_alcohol',
                        'intake_energy_kcal_intrinsic_sugar_protein',
                        'intake_energy_kcal_intrinsic_sugar_fat',
                        'intake_energy_kcal_intrinsic_sugar_alcohol'
                        )
                      ]
  
  # Calculate log
  selected_cols_log <- lapply(selected_cols, log)
  
  # Calculate z-score
  selected_cols_zscore <- lapply(selected_cols_log, scale)
  
  # Convert the list back to a data frame
  selected_cols_zscore_df <- as.data.frame(selected_cols_zscore)
  
  # Add the columns back to the original data frame
  names(selected_cols_zscore_df) <- paste0(names(selected_cols_zscore_df), "_std")
  df <- cbind(df, selected_cols_zscore_df)
  
  # Rename energy variables
  df <- rename(df, intake_energy_kcal_std_fiber = intake_energy_kcal_fiber_std)
  df <- rename(df, intake_energy_kcal_std_starch = intake_energy_kcal_starch_std)
  df <- rename(df, intake_energy_kcal_std_nmes = intake_energy_kcal_nmes_std)
  df <- rename(df, intake_energy_kcal_std_intrinsic_sugar = intake_energy_kcal_intrinsic_sugar_std)
  df <- rename(df, intake_energy_kcal_std_total_sugars = intake_energy_kcal_total_sugars_std)
  df <- rename(df, intake_energy_kcal_std_sucrose = intake_energy_kcal_sucrose_std)
  
  df <- rename(df, intake_energy_kcal_std_fiber_starch = intake_energy_kcal_fiber_starch_std)
  df <- rename(df, intake_energy_kcal_std_fiber_nmes = intake_energy_kcal_fiber_nmes_std)
  df <- rename(df, intake_energy_kcal_std_fiber_intrinsic_sugar = intake_energy_kcal_fiber_intrinsic_sugar_std)
  df <- rename(df, intake_energy_kcal_std_starch_nmes = intake_energy_kcal_starch_nmes_std)
  df <- rename(df, intake_energy_kcal_std_starch_intrinsic_sugar = intake_energy_kcal_starch_intrinsic_sugar_std)
  df <- rename(df, intake_energy_kcal_std_nmes_intrinsic_sugar = intake_energy_kcal_nmes_intrinsic_sugar_std)
  
  df <- rename(df, intake_energy_kcal_std_nmes_protein = intake_energy_kcal_nmes_protein_std)
  df <- rename(df, intake_energy_kcal_std_nmes_fat = intake_energy_kcal_nmes_fat_std)
  df <- rename(df, intake_energy_kcal_std_nmes_alcohol = intake_energy_kcal_nmes_alcohol_std)
  df <- rename(df, intake_energy_kcal_std_intrinsic_sugar_protein = intake_energy_kcal_intrinsic_sugar_protein_std)
  df <- rename(df, intake_energy_kcal_std_intrinsic_sugar_fat = intake_energy_kcal_intrinsic_sugar_fat_std)
  df <- rename(df, intake_energy_kcal_std_intrinsic_sugar_alcohol = intake_energy_kcal_intrinsic_sugar_alcohol_std)
  
  # Scale the bodyfat percentage
  df$bodyfat_percent_scale <- scale(df$bodyfat_percent)
  
  # Make Centre ID a factor
  df$centre_id <- factor(df$centre_id)
  
  # Make Gender a factor
  df$gender <- factor(df$gender)
  
  # Get one log variable for an outcome
  df$trig_log <- log(df$trig)
  
  df$intake_fiber_kcal_std <- df$intake_fibre_englyst_kcal_std
  df$intake_fiber_kcal <- df$intake_fibre_englyst_kcal
  df$intake_energy_kcal_std_intrinsic_sugars <- df$intake_energy_kcal_std_intrinsic_sugar
  df$trig_log <- log(df$trig)
  
  write.csv(df, paste0("../data/processed_files/subsetted_df_", df_name, ".csv"), row.names = FALSE)
}