# ==============================================================================
# PRAUC Metrics Generation
# ==============================================================================

#' Prepare SHAP Interaction Features
#'
#' @param int_path Full shap interaction features path
#' @param df.shap.totalMean Needed for id joining mapping
#' @return A configured dataframe ready for yardstick calculation
prepare_interaction_data <- function(int_path, data.for.RF) {
  library(dplyr)
  df.shap.full <- readr::read_csv(int_path, show_col_types = FALSE) %>%
    dplyr::select(Sample_ID, Repeat_ID:Pred_Cat, fileTag) %>%
    merge(unique(data.for.RF %>% dplyr::select(fileTag:MC_Ru)), by = 'fileTag')
    
  df.plot_curve <- df.shap.full %>%
    dplyr::mutate(
      MC_Ru = factor(MC_Ru, levels = c("1", "0"))
    )
    
  return(df.plot_curve)
}

#' Generate PR AUC Values Distribution
#'
#' @param df.plot_curve Configured interactions
#' @return Distributions grouping dataframe
calculate_prauc_distribution <- function(df.plot_curve) {
  library(dplyr)
  library(yardstick)
  
  df.auc_values <- df.plot_curve %>%
    dplyr::group_by(age, vowel, register, Repeat_ID) %>%
    yardstick::pr_auc(MC_Ru, Pred_Proba) %>%
    dplyr::ungroup()
    
  return(df.auc_values)
}

#' Calculate Precision-Recall Area Metrics
#'
#' @param df.plot_curve Standard interaction frame
#' @param df_lr_prep Logistic regression dataframe
#' @return Formatted precision-recall dataframes combining catboost and benchmark
calculate_pr_curves <- function(df.plot_curve, df_lr_prep) {
  library(dplyr)
  library(yardstick)
  
  pr_data <- df.plot_curve %>%
    dplyr::group_by(age, vowel, register, Repeat_ID) %>%
    yardstick::pr_curve(MC_Ru, Pred_Proba) %>%
    dplyr::ungroup()
    
  # Interpolate Points
  standard_recall <- seq(0, 1, length.out = 101)
  
  resampled_pr <- pr_data %>%
    dplyr::group_by(age, vowel, register, Repeat_ID) %>%
    dplyr::do({
      df_sub <- .
      interpolated_precision <- approx(x = df_sub$recall, y = df_sub$precision, xout = standard_recall, ties = max)$y
      data.frame(recall = standard_recall, precision = interpolated_precision)
    }) %>%
    dplyr::ungroup()
    
  # Standardize limits
  pr_summary <- resampled_pr %>%
    dplyr::group_by(age, vowel, register, recall) %>%
    dplyr::summarize(
      mean_precision = mean(precision, na.rm = TRUE),
      lower = min(precision, na.rm = TRUE),
      upper = max(precision, na.rm = TRUE),
      .groups = "drop"
    )
    
  # Logistic Regression Benchmark Extraction
  pr_baseline <- df_lr_prep %>%
    dplyr::mutate(MC_Ru = factor(MC_Ru, levels = c("1", "0"))) %>%
    dplyr::group_by(age, vowel, register) %>%
    yardstick::pr_curve(MC_Ru, prob_LR) %>%
    dplyr::do({
      df_sub <- .
      interpolated_precision <- approx(x = df_sub$recall, y = df_sub$precision, xout = standard_recall, ties = max)$y
      data.frame(recall = standard_recall, precision = interpolated_precision)
    }) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(Model = "LR (Baseline)")
    
  pr_main_model <- pr_summary %>%
    dplyr::rename(precision = mean_precision) %>%
    dplyr::mutate(Model = "CatBoost")
    
  # Merge Benchmark Values Together
  df_compare_plot <- dplyr::bind_rows(
    pr_main_model %>% dplyr::select(age, vowel, register, recall, precision, Model, lower, upper),
    pr_baseline %>% dplyr::mutate(lower = NA, upper = NA)
  )
  
  return(df_compare_plot)
}

#' Calculate Individual Participant Subject Analysis
#'
#' @param df.plot_curve Interactions metrics list 
#' @return Group aggregated stats dataframe
calculate_individual_prauc <- function(df.plot_curve) {
  library(dplyr)
  library(purrr)
  library(yardstick)
  
  safe_pr_auc <- purrr::possibly(
    .f = function(data, truth, estimate) { yardstick::pr_auc(data, !!rlang::enquo(truth), !!rlang::enquo(estimate)) },
    otherwise = tibble::tibble(.metric = "pr_auc", .estimator = "binary", .estimate = NA_real_)
  )
  
  df.subject_run_auc <- df.plot_curve %>%
    dplyr::group_by(id, age, vowel, register, Repeat_ID) %>%
    dplyr::do(safe_pr_auc(., truth = MC_Ru, estimate = Pred_Proba)) %>%
    dplyr::ungroup()
    
  df.subject_final <- df.subject_run_auc %>%
    dplyr::group_by(id, age, vowel, register) %>%
    dplyr::summarize(mean_auc = mean(.estimate, na.rm = TRUE), .groups = "drop")
    
  return(df.subject_final)
}