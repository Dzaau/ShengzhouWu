# ==============================================================================
# Model Stability and Explanation Consistency Analysis
# ==============================================================================

#' Analyze Performance Stability (AUCPR)
#'
#' @param df_perf The dataframe of performance results
#' @return Expected summary statistics and Friedman test results
analyze_performance_stability <- function(df_perf) {
  library(dplyr)
  library(rstatix)
  
  # Calculate summary statistics: Mean, SD, 95% CI, and CV
  perf_summary <- df_perf %>%
    dplyr::summarize(
      Mean_AUCPR = mean(AUCPR),
      SD_AUCPR = sd(AUCPR),
      SE_AUCPR = sd(AUCPR) / sqrt(dplyr::n()),
      CI_Lower = Mean_AUCPR - qt(0.975, df=dplyr::n()-1) * SE_AUCPR,
      CI_Upper = Mean_AUCPR + qt(0.975, df=dplyr::n()-1) * SE_AUCPR,
      CV_Percent = (SD_AUCPR / Mean_AUCPR) * 100
    )
  
  cat("--- Performance Summary ---\n")
  print(perf_summary)
  
  # Non-parametric Friedman test
  # Block/subject: Fold, Treatment/group: Rep
  cat("\n--- Friedman Test for AUCPR across Repetitions ---\n")
  friedman_res <- df_perf %>% rstatix::friedman_test(AUCPR ~ Rep | Fold)
  print(friedman_res)
  
  friedman_res_1 <- df_perf %>% rstatix::friedman_test(AUCPR ~ Fold | Rep)
  print(friedman_res_1)
  
  return(list(summary = perf_summary, test_rep = friedman_res, test_fold = friedman_res_1))
}

#' Preprocess SHAP Ranks
#'
#' @param df_shap Raw SHAP dataframe
#' @return A list containing Tables A and B for dimension calculations, and Wide_SHAP
preprocess_shap_ranks <- function(df_shap) {
  library(dplyr)
  library(tidyr)
  
  df_shap_wide <- df_shap %>%
    dplyr::mutate(Feature_Name = paste0("shap_", Feature)) %>%
    dplyr::select(Sample_ID, Rep, Fold, Feature_Name, SHAP_Value) %>%
    tidyr::pivot_wider(names_from = Feature_Name, values_from = SHAP_Value)
  
  # Table A: Average across all folds and samples for each Rep
  table_a <- df_shap %>%
    dplyr::group_by(Rep, Feature) %>%
    dplyr::summarize(Mean_Abs_SHAP = mean(abs(SHAP_Value)), .groups = "drop") %>%
    dplyr::group_by(Rep) %>%
    dplyr::mutate(Rank = dplyr::row_number(desc(Mean_Abs_SHAP))) %>% # Safely ranks
    dplyr::ungroup()
  
  # Table B: Average across all samples for each Fold within a typical Rep (e.g., Rep 1)
  table_b <- df_shap %>%
    dplyr::filter(Rep == 1) %>% # Choose a typical Rep
    dplyr::group_by(Fold, Feature) %>%
    dplyr::summarize(Mean_Abs_SHAP = mean(abs(SHAP_Value)), .groups = "drop") %>%
    dplyr::group_by(Fold) %>%
    dplyr::mutate(Rank = dplyr::row_number(desc(Mean_Abs_SHAP))) %>%
    dplyr::ungroup()
    
  return(list(Table_A = table_a, Table_B = table_b, Wide_SHAP = df_shap_wide))
}

#' Calculate Kendall's W Dual-Dimension Stability Test
#'
#' @param table_a Dimension A table
#' @param table_b Dimension B table
#' @return W test results list
calculate_kendall_w <- function(table_a, table_b) {
  library(dplyr)
  library(tidyr)
  library(irr)
  
  # Dimension A: Repetitions
  mat_a <- table_a %>%
    dplyr::select(Rep, Feature, Rank) %>%
    tidyr::pivot_wider(names_from = Rep, values_from = Rank) %>%
    tibble::column_to_rownames("Feature") %>%
    as.matrix()
    
  # Dimension B: Folds
  mat_b <- table_b %>%
    dplyr::select(Fold, Feature, Rank) %>%
    tidyr::pivot_wider(names_from = Fold, values_from = Rank) %>%
    tibble::column_to_rownames("Feature") %>%
    as.matrix()
  
  w_res_a <- irr::kendall(mat_a, correct = TRUE)
  w_res_b <- irr::kendall(mat_b, correct = TRUE)
  
  cat("--- Kendall's W for Repetitions (Algorithm Stability) ---\n")
  print(w_res_a)
  cat("\n--- Kendall's W for Folds (Data Perturbation Stability) ---\n")
  print(w_res_b)
  
  return(list(W_Reps = w_res_a, W_Folds = w_res_b))
}