# ==============================================================================
# Overall SHAP Importance Calculations
# ==============================================================================

#' Identify Groups
#' 
#' @return The standard feature_to_group dataframe for the analysis
get_group_lists <- function() {
  group_list <- list(
    Pitch = c('pitch_change', 'pitch_height'),
    Formant = c('F1_z', 'F2_z'),
    Voice = c("SQ_z", "CQ_z", 
              "voice_PC1", "voice_PC2", "voice_PC4", 
              "voice_PC6", "voice_PC7", "voice_PC8", "voice_PC9", 
              "creak"),
    Coda = c('glottalStop'),
    Duration = c('duration_z')
  )
  
  feature_to_group <- data.frame(
    feature = unlist(group_list),
    group = rep(names(group_list), times = lengths(group_list)),
    stringsAsFactors = FALSE
  )
  return(list(list=group_list, df=feature_to_group))
}

#' Calculate Feature Importance with Bootstrapped CI
#'
#' @param df.shap.total20 Aggregated shap dataframe
#' @param feature_to_group Group mapping
#' @param n_reps Bootstrapping repetitions
#' @return A dataframe holding group metrics, importances and CIs
calculate_shap_importance <- function(df.shap.total20, feature_to_group, n_reps = 1000) {
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(boot)
  
  shap_cols <- colnames(df.shap.total20)[grep("^shap_", colnames(df.shap.total20))]
  raw_feats <- sub("^shap_", "", shap_cols)
  
  # Calculate base importance
  feature_importance <- df.shap.total20 %>%
    dplyr::select(dplyr::all_of(shap_cols)) %>%
    tidyr::pivot_longer(cols = dplyr::everything(), names_to = "shap_col", values_to = "shap_val") %>%
    dplyr::mutate(feature = sub("^shap_", "", shap_col)) %>%
    dplyr::group_by(feature) %>%
    dplyr::summarise(importance = mean(abs(shap_val)), .groups="drop") %>%
    dplyr::left_join(feature_to_group, by = "feature") %>%
    dplyr::mutate(group = tidyr::replace_na(group, "Other")) %>%
    dplyr::arrange(desc(importance))
    
  # Bootstrapping for CI
  boot_mean <- function(data, indices) { return(mean(abs(data[indices]))) }
  ci_list <- list()
  target_prefix <- "shap_"
  
  # Select only columns starting with target prefix
  cols_to_process <- names(df.shap.total20)[startsWith(names(df.shap.total20), target_prefix)]
  
  for (col in cols_to_process) {
    boot_out <- boot(data = df.shap.total20[[col]], statistic = boot_mean, R = n_reps)
    ci_res <- boot.ci(boot_out, type = "perc")
    clean_name <- stringr::str_remove(col, paste0("^", target_prefix))
    
    # Store
    ci_list[[col]] <- data.frame(
      feature = clean_name,
      CI_Lower = ci_res$percent[4],
      CI_Upper = ci_res$percent[5]
    )
  }
  
  df_ci_all <- dplyr::bind_rows(ci_list)
  feature_importance <- feature_importance %>%
    dplyr::left_join(df_ci_all, by = "feature")
    
  # Factorize with proper arrangement
  feature_order <- feature_importance$feature[order(feature_importance$importance)]
  feature_importance$feature <- as.factor(feature_importance$feature)
  feature_importance$feature <- factor(feature_importance$feature, levels = feature_order)
  
  return(list(feat_imp = feature_importance, raw_feats = raw_feats))
}


#' Calculate Trend Lines
#'
#' @param df Raw aggregated data object
#' @param feature_importance Output from calculate_shap_importance
#' @param raw_feats List of features
#' @return Extracted dataframe matching X and Y loess predictions for trends
calculate_trend_lines <- function(df, feature_importance, raw_feats) {
  library(dplyr)
  trend_data_list <- list()
  
  x_scale_factor <- max(feature_importance$importance) * 0.35 
  y_scale_factor <- 0.35
  
  for (feat in raw_feats) {
    if (is.numeric(df[[feat]])) {
      x_val <- df[[feat]]
      y_val <- df[[paste0("shap_", feat)]]
      plot_data <- data.frame(x = x_val, y = y_val)
      
      x_norm <- (plot_data$x - min(plot_data$x)) / (max(plot_data$x) - min(plot_data$x))
      x_final <- x_norm * x_scale_factor
      
      model_loess <- loess(y ~ x, data = data.frame(x = x_final, y = y_val), span = 0.75)
      
      x_pred <- seq(min(x_final), max(x_final), length.out = 50)
      y_pred <- predict(model_loess, newdata = data.frame(x = x_pred))
      
      # Reverse order mapped
      feat_idx <- nrow(feature_importance) + 1 - which(feature_importance$feature == feat)
      y_final <- feat_idx + (y_pred / max(abs(y_pred))) * y_scale_factor
      
      trend_data_list[[feat]] <- data.frame(
        feature = feat,
        x_plot = x_pred,
        y_plot = y_final,
        group = feature_importance$group[feature_importance$feature == feat]
      )
    }
  }
  
  trend_df <- dplyr::bind_rows(trend_data_list)
  return(trend_df)
}


#' Calculate Rose Data (Sector Width/Angles)
#' 
#' @param feature_importance Output dataframe
#' @return Rose mapping dataset with polar coordinates pre-calculated
prepare_rose_data <- function(feature_importance) {
  library(dplyr)
  rose_data_processed <- feature_importance %>%
    dplyr::arrange(group, importance) %>%
    dplyr::mutate(id = dplyr::row_number()) %>%
    dplyr::arrange(desc(importance)) %>%
    dplyr::mutate(
      rel_width = importance / sum(importance), 
      xmax = cumsum(rel_width),
      xmin = xmax - rel_width,
      x_center = (xmin + xmax) / 2,
      ord = order(importance, decreasing = TRUE)
    )
  return(rose_data_processed)
}


#' Calculate Group Importance Means
#'
#' @param df The data
#' @param feature_to_group Group relationships
#' @return The grouped evaluation scores
calculate_group_importance <- function(df, feature_to_group) {
  library(dplyr)
  
  group_imp_df <- data.frame()
  unique_groups <- unique(feature_to_group$group)
  
  for (grp in unique_groups) {
    grp_feats <- feature_to_group$feature[feature_to_group$group == grp]
    grp_shap_cols <- paste0("shap_", grp_feats)
    
    if (length(grp_shap_cols) > 0) {
      valid_cols <- intersect(grp_shap_cols, colnames(df))
      if (length(valid_cols) > 0) {
        row_sums <- rowSums(df[, valid_cols, drop = FALSE])
        grp_imp <- mean(abs(row_sums))
        group_imp_df <- rbind(group_imp_df, data.frame(group = grp, importance = grp_imp))
      }
    }
  }
  
  group_imp_df <- group_imp_df %>% dplyr::arrange(importance)
  group_imp_df$group <- factor(group_imp_df$group, levels = group_imp_df$group)
  
  return(group_imp_df)
}