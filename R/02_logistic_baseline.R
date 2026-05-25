# ==============================================================================
# Logistic Baseline Model
# ==============================================================================

#' Train Logistic Regression Baseline
#'
#' @param df_lr_prep Dataframe prepared by prepare_lr_prep_data
#' @return A list containing the fitted glm model and the original dataframe with probabilities
train_logistic_baseline <- function(df_lr_prep) {
  library(dplyr)
  
  lr_model <- glm(y ~ ., 
                  data = df_lr_prep %>% dplyr::select(-age, -vowel, -register, -MC_Ru), 
                  family = binomial)
                  
  # Add predictions to dataset
  df_lr_prep$prob_LR <- predict(lr_model, newdata = df_lr_prep, type = "response")
  
  return(list(
    model = lr_model,
    df_with_preds = df_lr_prep
  ))
}