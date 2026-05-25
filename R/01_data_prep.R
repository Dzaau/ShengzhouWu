# ==============================================================================
# Data Preparation
# ==============================================================================

#' Load Base Data
#'
#' Load data.for.ML if not already in the environment.
#' @param data_path The path to the file containing data.for.ML
#' @return The data.for.ML dataset.
load_base_data <- function(data_path = NULL) {
  if (exists("data.for.ML")) {
    return(data.for.ML)
  }
  if (!is.null(data_path)) {
    # If it's a joblib or similar, user should provide the generic R frame
    # Assuming user loads data.for.ML beforehand or reading a specific RDS/csv
    # Placeholder for actual loading logic
    message("data.for.ML not found in environment. Returning NULL.")
    return(NULL) 
  }
}

#' Prepare SHAP and Performance Data
#'
#' @param data_dir The directory containing generated CSV data
#' @param base_data The data.for.ML object
#' @return A list with preprocessed data frames
prepare_shap_and_performance_data <- function(data_dir, base_data) {
  library(dplyr)
  library(tidyr)
  library(stringr)
  
  perf_path <- file.path(data_dir, "df_performance.csv")
  shap_path <- file.path(data_dir, "df_raw_shap.csv")
  
  if (!file.exists(perf_path) || !file.exists(shap_path)) {
    stop("Data files not found. Please run generate_data.R first.")
  }
  
  # Load Performance
  df_performance <- readr::read_csv(perf_path, show_col_types = FALSE) %>%
    dplyr::select(Rep = Repeat_ID, Fold = Fold_ID, AUCPR = PR_AUC) %>%
    dplyr::mutate(Rep = as.factor(Rep), Fold = as.factor(Fold))
  
  # Load SHAP
  df_raw_shap <- readr::read_csv(shap_path, show_col_types = FALSE) %>%
    dplyr::filter(!stringr::str_detect(Feature, "Path")) %>%
    dplyr::mutate(
      Rep = as.factor(Rep),
      Fold = as.factor(Fold),
      SHAP_Value = as.numeric(SHAP_Value)
    )
  
  # Process wide SHAP
  df.trueSHAP <- df_raw_shap %>%
    dplyr::mutate(Feature_Name = paste0("shap_", Feature)) %>%
    dplyr::select(Sample_ID, Rep, Fold, Feature_Name, SHAP_Value) %>%
    tidyr::pivot_wider(names_from = Feature_Name, values_from = SHAP_Value) %>%
    dplyr::mutate(fileTag = base_data$fileTag[Sample_ID + 1]) %>%
    merge(base_data, by = 'fileTag')
  
  # Total SHAP 20
  df.shap.total20 <- df.trueSHAP %>%
    dplyr::select(Rep, where(is.numeric)) %>% 
    group_by(Rep) %>%
    dplyr::summarize(
      across(where(is.numeric), ~ mean(abs(.x), na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    dplyr::select(Rep, shap_sex:shap_ref) 
  
  # Total SHAP Mean
  df.shap.totalMean <- df.trueSHAP %>%
    dplyr::select(fileTag, where(is.numeric)) %>%
    group_by(fileTag) %>%
    dplyr::summarize(
      across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    dplyr::mutate(edu = as.factor(edu))
    
  return(list(
    df_performance = df_performance,
    df_raw_shap = df_raw_shap,
    df.trueSHAP = df.trueSHAP,
    df.shap.total20 = df.shap.total20,
    df.shap.totalMean = df.shap.totalMean
  ))
}

#' Prepare Data for Baseline Logistic Regression
#'
#' @param base_data The data.for.ML dataframe
#' @return A processed dataframe for baseline model training
prepare_lr_prep_data <- function(base_data) {
  library(dplyr)
  include_vars <- c(
    "YearOfBirth", "sex", "edu", "dominance", "condition",
    "glottalStop", "pitch_height", "pitch_change", "creak",
    "CQ_z", "SQ_z", "voice_PC1", "voice_PC2", "voice_PC4",
    "voice_PC6", "voice_PC7", "voice_PC8", "voice_PC9",
    "duration_z", "F1_z", "F2_z"
  )
  
  df_lr_prep <- base_data %>%
    dplyr::select(MC_Ru, dplyr::all_of(include_vars), age, vowel, register) %>%
    na.omit() %>%
    dplyr::mutate(
      y = as.numeric(as.character(MC_Ru)),
      sex = as.factor(sex),
      condition = as.factor(condition),
      edu = as.factor(edu)
    )
  return(df_lr_prep)
}
