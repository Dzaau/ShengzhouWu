# ==============================================================================
# Bayesian Modeling and Feature Analysis
# ==============================================================================

#' Prepare Bayesian Data
#'
#' @param df.shap.everyToken Generated aggregations
#' @return Formatted wide matrix configured for bayesian predictions
prepare_bayesian_data <- function(df.trueSHAP) {
  library(dplyr)
  library(tidyr)
  
  df.shap.everyToken = df.trueSHAP %>%
    group_by(fileTag) %>%
    dplyr::summarize(across(everything(), ~ if(is.numeric(.)){ mean(., na.rm = TRUE)} else{first(.)}), .groups = "drop") %>%
    dplyr::select(-Sample_ID:-Rep) %>%
    dplyr::mutate(Pitch = abs(shap_pitch_height + shap_pitch_change),
           Formant = abs(shap_F1_z + shap_F2_z),
           Duration = abs(shap_duration_z),
           Voice = abs(shap_voice_PC1 + shap_voice_PC2 + shap_voice_PC4 + shap_voice_PC6 + 
                         shap_voice_PC7 + shap_voice_PC8 + shap_voice_PC9 + 
                         shap_CQ_z + shap_SQ_z + shap_creak),
           Coda = abs(shap_glottalStop)) %>%
    tidyr::pivot_longer(
      cols = Pitch:Coda,  # 最后6列
      names_to = "Feature",
      values_to = "importance"
    ) %>%
    dplyr::mutate(Feature = factor(Feature, 
                            levels = c("Coda", "Pitch", "Voice", "Formant","Duration"))) %>%
    group_by(fileTag) %>%
    # 2. 计算相对权重
    dplyr::mutate(
      # 计算该 trial 内所有特征的总拉力
      total_trial_shap = sum(importance),
      # 计算相对权重 (0 到 1 之间)
      relative_weight = importance / total_trial_shap
    ) %>%
    # 3. 解除分组
    ungroup() %>%
    group_by(fileTag,Feature) %>%
    dplyr::summarize(across(everything(), ~ if(is.numeric(.)){ mean(., na.rm = TRUE)} else{first(.)}), .groups = "drop") %>%
    dplyr::select(-shap_sex:-shap_ref,-creak:-voice_PC9) 
  
  # Format base structures relative weights
  data_wide <- df.shap.everyToken %>%
    dplyr::select(fileTag, id, age, vowel, register, Feature, relative_weight) %>%
    tidyr::pivot_wider(names_from = Feature, values_from = relative_weight) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      total = sum(dplyr::c_across(Coda:Duration)),
      dplyr::across(Coda:Duration, ~ . / total)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      vowel = as.factor(vowel),
      age = as.factor(age),
      register = as.factor(register),
      id = as.factor(id),
      VR = interaction(vowel, register, sep = "_")
    )
    
  features <- c("Coda", "Pitch", "Voice", "Formant", "Duration")
  Y_matrix <- as.matrix(data_wide[, features])
  data_wide$Y <- Y_matrix
  
  return(data_wide)
}

#' Generate and Compile Bayesian Model
#'
#' @param data_wide The structural formatted set
#' @param mode 'id' or 'age' configuration selector
#' @param force_retrain Boolean controlling compilation
#' @param output_dir Working dir where cache might reside
#' @return A resolved loaded BRMS parameter stack
compile_bayesian_model <- function(data_wide, mode = 'id', force_retrain = FALSE, output_dir = ".") {
  library(brms)
  library(cmdstanr)
  
  file_name <- paste0("bayes_catBoost_", mode, ".RData")
  model_path <- file.path(output_dir, file_name)
  
  if (!force_retrain && file.exists(model_path)) {
    message("Loading existing model: ", file_name)
    load(model_path)
    if (mode == 'id') return(fit_dirichlet_id)
    if (mode == 'age') return(fit_dirichlet_age)
  }
  
  message("Training Bayesian Mode: ", mode)
  
  my_priors <- c(
    prior(normal(0, 5), class = "b", dpar = 'muDuration'),
    prior(normal(0, 5), class = "b", dpar = 'muFormant'),
    prior(normal(0, 5), class = "b", dpar = 'muVoice'),
    prior(normal(0, 5), class = "b", dpar = 'muPitch')
  )
  
  if (mode == "age") {
    form <- bf(Y ~ 0 + age:vowel:register + (1 | id))
  } else {
    form <- bf(Y ~ 0 + VR + (0 + VR | id))
  }
  
  fit <- brm(
    formula = form,
    data = data_wide,
    family = dirichlet(link = "logit"), 
    prior = my_priors,
    chains = 4, init = 0, iter = 2000, warmup = 1000, cores = 4, seed = 123,
    refresh = 50, backend = "cmdstanr"
  )
  
  # Save dynamically
  if (mode == "id") {
    fit_dirichlet_id <- fit
    save(fit_dirichlet_id, file = model_path)
  } else {
    fit_dirichlet_age <- fit
    save(fit_dirichlet_age, file = model_path)
  }
  
  return(fit)
}

#' Extract Group Differential Effects 
#'
#' @param fit_dirichlet_age Extracted structure
#' @param data_wide Parameter sets
#' @return Group analysis dataframe configuration differences resolving
extract_bayesian_group_effects <- function(fit_dirichlet_age, data_wide) {
  library(dplyr)
  library(tidybayes)
  library(modelr)
  library(marginaleffects)
  
  conditions <- data_wide %>% modelr::data_grid(age, vowel, register)
  
  posterior_proportions <- fit_dirichlet_age %>%
    add_epred_draws(newdata = conditions, re_formula = NA)
    
  summary_diffs <- posterior_proportions %>%
    dplyr::ungroup() %>%
    dplyr::select(-.row:-.iteration) %>% 
    tidyr::pivot_wider(names_from = age, values_from = .epred) %>%
    dplyr::mutate(diff = Young - Old) %>%
    dplyr::group_by(vowel, register, .category) %>%
    median_qi(diff, .width = 0.90) %>%
    dplyr::mutate(
      is_sig = dplyr::if_else(.lower * .upper > 0, "Yes", "No"),
      Feature = .category
    )
    
  preds_age <- marginaleffects::predictions(
    fit_dirichlet_age, 
    newdata = marginaleffects::datagrid(vowel = unique, register = unique, age = unique),
    type = "response"
  ) %>%
    dplyr::mutate(Feature = factor(group, levels = c("Coda", "Pitch", "Voice", "Formant", "Duration"))) %>%
    merge(summary_diffs %>% dplyr::select(vowel, register, is_sig, Feature), by = c('vowel', 'register', 'Feature'))
    
  feature_levels <- c("Coda", "Pitch", "Voice", "Formant", "Duration")
  preds_age$Feature <- factor(preds_age$Feature, levels = feature_levels)
  
  sig_rows_age <- preds_age %>%
    mutate(y_index = as.numeric(factor(Feature, levels = feature_levels))) %>%
    filter(is_sig == "Yes") %>%
    select(Feature, register, vowel, y_index) %>%
    distinct()
  
  return(list(preds_age = preds_age, sig_rows_age = sig_rows_age, feature_levels = feature_levels))
}

#' Calculate Clustering Configurations
#'
#' @param fit_dirichlet_id Output bayes structure
#' @param data_wide Original formats
#' @param data.for.RF Origin database resolving id dimensions
#' @return Structural clusters dataset configurations
calculate_clustering <- function(fit_dirichlet_id, data_wide, data.for.RF) {
  library(dplyr)
  library(tidyr)
  library(marginaleffects)
  library(NbClust)
  library(factoextra)
  library(tibble)
  
  subject_preds_id <- marginaleffects::predictions(
    fit_dirichlet_id,
    newdata = marginaleffects::datagrid(id = unique(data_wide$id), VR = unique(data_wide$VR)),
    re_formula = NULL, type = "response"
  )
  
  subject_preds_clean_id <- subject_preds_id %>%
    as.data.frame() %>%
    dplyr::mutate(Feature = group) %>%
    tidyr::extract(VR, into = c('vowel', 'register'), regex = "^(.*)_(.*)")
    
  # CLR Mapping Functions
  apply_clr <- function(x) { log_x <- log(x); return(log_x - mean(log_x)) }
  
  clustering_data_CLR <- subject_preds_clean_id %>%
    dplyr::group_by(id, vowel, register) %>%
    dplyr::mutate(clr_estimate = apply_clr(estimate)) %>%
    dplyr::ungroup() %>%
    tidyr::pivot_wider(id_cols = id, names_from = c(vowel, register, Feature), values_from = clr_estimate) %>%
    tibble::column_to_rownames("id")
    
  nb_res <- NbClust(clustering_data_CLR, 
                    distance = "euclidean", 
                    min.nc = 2, max.nc = 6, 
                    method = "ward.D2", 
                    index = "silhouette")
  # Optimal K
  best_k <- which.max(nb_res$All.index) + 1
  print(paste("Best K:", best_k))
  
  dist_mat <- dist(clustering_data_CLR, method = "euclidean")
  hc <- hclust(dist_mat, method = "ward.D2")
  clusters <- cutree(hc, k = best_k)
  
  # PCA mapping formats merges
  final_results <- data.frame(id = rownames(clustering_data_CLR), cluster = clusters) %>%
    merge(clustering_data_CLR %>% dplyr::mutate(id = rownames(clustering_data_CLR)), by = 'id') %>%
    merge(unique(data.for.RF %>% dplyr::select(id, dominance, YearOfBirth)), by ='id')

  data.pca <- final_results %>% select(-id, -cluster, -YearOfBirth, -dominance)
  
  rownames(data.pca) <- final_results$id
  res.pca <- prcomp(data.pca, scale. = FALSE)
  
  pc_data <- res.pca$x[, 1:5]
  
  df.catBoost.pca <- as.data.frame(pc_data) %>%
    dplyr::mutate(id = rownames(pc_data)) %>%
    merge(final_results %>% dplyr::select(id, YearOfBirth, dominance, cluster), by='id') %>%
    mutate(cluster = case_when(
      cluster == 2 ~ 3,
      cluster == 3 ~ 2,
      TRUE ~ cluster
      ))
    
  return(list(
    pred_id = subject_preds_clean_id,
    clustering_data = clustering_data_CLR,
    hc = hc,
    nb_res = nb_res,
    pca = res.pca,
    df_pca = df.catBoost.pca
  ))
}