
# ------------------------------------------------------------------------------
# 1. Package Initialization
# ------------------------------------------------------------------------------
# Suppress package startup messages for a cleaner console output
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(boot)
  library(rstatix)
  library(irr)
  library(yardstick)
  library(brms)
  library(modelr)
  library(marginaleffects)
  library(tidybayes)
  library(bayestestR)
  library(cmdstanr)
  library(factoextra)
  library(NbClust)
  library(pheatmap)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
  library(dendextend)
  library(ggplot2)
  library(ggridges)
  library(ggrepel)
  library(RColorBrewer)
  library(patchwork)
  library(caret)
  library(pROC)
  library(PRROC)
  library(tidyverse)
})

# ------------------------------------------------------------------------------
# 2. Source Sub-Modules
# ------------------------------------------------------------------------------
# Assume working directory is set to 'CatBoostR_Cleaned'
setwd('D:/Academic/shengzhou/ML/CatBoostR_Cleaned')
source("01_data_prep.R")
source("02_logistic_baseline.R")
source("03_stability.R")
source("04_overall_shap.R")
source("05_prauc.R")
source("06_bayesian_models.R")
load("dataForML.RData")

# ------------------------------------------------------------------------------
# 3. Parameters & Paths
# ------------------------------------------------------------------------------
# Modify these paths accordingly
data_workspace    <- "d:/Academic/shengzhou/ML/test/catboost_grouped_20260409"
interactions_path <- file.path(data_workspace, "full_shap_interaction_features.csv")
force_retrain     <- FALSE  # Set to TRUE to force retraining Bayesian models

# Base dataset loading dummy placeholder. Let's assume user loads this manually or via RDS
# using readRDS("path_to_data.rds")
# If it's already in the global environment, load_base_data() captures it.
base_data <- load_base_data()
if (is.null(base_data)) {
  stop("Missing base data 'data.for.ML'. Please load it into the environment or pass path constraints.")
}

# ------------------------------------------------------------------------------
# 4. Pipeline Execution
# ------------------------------------------------------------------------------
cat(">>> Starting Data Preparation Pipeline...\n")
prep_res       <- prepare_shap_and_peMLormance_data(data_workspace, base_data)
df_lr_prep     <- prepare_lr_prep_data(base_data)

cat(">>> Running Logistic Baseline Benchmark...\n")
lr_res         <- train_logistic_baseline(df_lr_prep)

cat(">>> Executing Stability Analysis...\n")
stab_peML      <- analyze_peMLormance_stability(prep_res$df_peMLormance)
stab_shap      <- preprocess_shap_ranks(prep_res$df_raw_shap)
kendall_res    <- calculate_kendall_w(stab_shap$Table_A, stab_shap$Table_B)

cat(">>> Processing Global SHAP Importance...\n")
group_metrics  <- get_group_lists()
shap_imp_res   <- calculate_shap_importance(prep_res$df.shap.total20, group_metrics$df, n_reps = 1000)
trend_df       <- calculate_trend_lines(prep_res$df.shap.totalMean, shap_imp_res$feat_imp, shap_imp_res$raw_feats)
rose_data      <- prepare_rose_data(shap_imp_res$feat_imp)
group_imp_df   <- calculate_group_importance(prep_res$df.shap.totalMean, group_metrics$df)

cat(">>> Computing PRAUC Metrics...\n")
df.plot_curve  <- prepare_interaction_data(interactions_path, data.for.ML)
pr_dist        <- calculate_prauc_distribution(df.plot_curve)
pr_compare     <- calculate_pr_curves(df.plot_curve, lr_res$df_with_preds)
ind_pr_auc     <- calculate_individual_prauc(df.plot_curve)

cat(">>> Executing Bayesian Dirichlet Inference & PCA...\n")
bay_data       <- prepare_bayesian_data(prep_res$df.trueSHAP)
bay_id_model   <- compile_bayesian_model(bay_data, mode = 'id', force_retrain, output_dir = ".")
bay_age_model  <- compile_bayesian_model(bay_data, mode = 'age', force_retrain, output_dir = ".")

bay_group_eff  <- extract_bayesian_group_effects(bay_age_model, bay_data)
cluster_res    <- calculate_clustering(bay_id_model, bay_data, base_data)

# ------------------------------------------------------------------------------
# 5. Statistical Analysis
# ------------------------------------------------------------------------------

# SHAP Wilcoxon signed-rank test

df_ref <- stab_shap$Table_A %>%
  filter(Feature == "ref") %>%
  select(Rep, Ref_SHAP = Mean_Abs_SHAP)

shap_test_results <- stab_shap$Table_A %>%
  filter(Feature != "ref") %>%
  left_join(df_ref, by = "Rep") %>%
  group_by(Feature) %>%
  summarise(
    Mean_Feature = mean(Mean_Abs_SHAP),
    Mean_Ref = mean(Ref_SHAP),
    Mean_Diff = Mean_Feature - Mean_Ref,
    test = list(wilcox.test(Mean_Abs_SHAP, Ref_SHAP, paired = TRUE, exact = FALSE))
  ) %>%
  mutate(
    V_stat = map_dbl(test, ~ .x$statistic),
    p_raw = map_dbl(test, ~ .x$p.value)
  ) %>%
  select(-test) %>%
  mutate(
    p_adj = p.adjust(p_raw, method = "bonferroni"),
    Sig = case_when(
      p_adj < 0.001 ~ "***",
      p_adj < 0.01  ~ "**",
      p_adj < 0.05  ~ "*",
      TRUE          ~ "ns"
    )
  ) %>%
  arrange(p_adj)


# Correlations between cue weights and birth year/dominance

df.cor.analysis <- cor.dimensions %>%
  mutate(YoB = scale(YearOfBirth)[,1],
         dom = scale(dominance)[,1]) %>%
  pivot_longer(
    cols = contains("_"), 
    names_to = c("vowel", "register", "feature"),
    names_sep = "_",
    values_to = "weight"
  ) %>%
  pivot_longer(
    cols = c(YoB, dom),
    names_to = "target_var",
    values_to = "target_value"
  )

cor_results <- df.cor.analysis %>%
  group_by(vowel, register, feature, target_var) %>%
  summarise(
    res = list(cor.test(weight, target_value, method = "pearson")),
    .groups = "drop"
  ) %>%
  mutate(
    r = map_dbl(res, ~ .x$estimate),
    p_value = map_dbl(res, ~ .x$p.value),
    sig = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE ~ ""
    )
  ) %>%
  mutate(feature = factor(feature,levels = c('Coda','Pitch','Voice','Formant','Duration')))


# LR metrics

df.LR = lr_res$df_with_preds

df.LR$MC_Ru <- as.factor(df.LR$MC_Ru)
df.LR$y <- as.factor(ifelse(df.LR$prob_LR >= 0.5, 1, 0))

cm <- confusionMatrix(df.LR$y, df.LR$MC_Ru, positive = "1")

print(cm$table)
stats <- cm$byClass
accuracy <- cm$overall['Accuracy']

roc_obj <- roc(df.LR$MC_Ru, df.LR$prob_LR)
roc_auc <- auc(roc_obj)

fg <- df.LR$prob_LR[df.LR$MC_Ru == 1]
bg <- df.LR$prob_LR[df.LR$MC_Ru == 0]

pr_obj <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
pr_auc <- pr_obj$auc.integral

## Group-level Bayesian model

load("bayes_catBoost_age.RData")

formatted_ranking <- predictions(
  fit_dirichlet_age,
  newdata = datagrid(age = unique, vowel = unique, register = unique),
  type = "response"
) %>%
  as.data.frame() %>%
  mutate(
    est_fmt = sprintf("%.3f", estimate),
    low_fmt = sprintf("%.3f", conf.low),
    high_fmt = sprintf("%.3f", conf.high)
  ) %>%
  mutate(feature_info = paste0(group, " ",est_fmt, " [", low_fmt, ", ", high_fmt, "]")) %>%
  group_by(age, vowel, register) %>%
  arrange(desc(estimate), .by_group = TRUE) %>%
  mutate(rank = row_number()) %>%
  ungroup() %>%
  select(age, vowel, register, rank, feature_info) %>%
  pivot_wider(
    names_from = rank, 
    values_from = feature_info,
    names_prefix = ""
  )

# Different CrI levels

conditions <- bay_data %>% modelr::data_grid(age, vowel, register)

posterior_proportions <- fit_dirichlet_age %>%
  add_epred_draws(newdata = conditions, re_formula = NA)

age_diffs <- posterior_proportions %>%
  dplyr::ungroup() %>%
  dplyr::select(-.row:-.iteration) %>% 
  tidyr::pivot_wider(names_from = age, values_from = .epred) %>%
  dplyr::mutate(diff = Young - Old)

summary_diffs <- age_diffs %>%
  dplyr::group_by(vowel, register, .category) %>%
  mean_qi(diff, .width = c(0.95, 0.85, 0.75)) %>%
  dplyr::mutate(
    hdi0 = dplyr::if_else(.lower * .upper > 0, "No", "Yes"),
    Feature = .category
  )


# ------------------------------------------------------------------------------
# 5. Visualization
# ------------------------------------------------------------------------------
theme_my_style <- function(base_family = "sans") {
  theme_bw(base_family = base_family)+
    theme(strip.text = element_text(size = 12,hjust=0.5,vjust = 1, face = "italic",
                                    margin = margin(t = 5, b = 5, l=5, r=5, unit = "pt")),
          strip.background = element_rect(
            color="black", fill='grey95', linewidth=0.5, linetype="solid"
          ),
          plot.title = element_text(size = 14, face = "bold"),
          axis.title.x = element_text(size = 12, face = "bold"),
          axis.title.y = element_text(size = 12, face = "bold", angle = 90),
          axis.text = element_text(size = 12),
          legend.title = element_text(size = 12, face = "bold"),
          legend.text = element_text(size = 12),
          legend.position = "none",
          panel.border = element_rect(colour = "black", linewidth = 0.5),
          panel.spacing.x = unit(10, "pt"),  # 列间距
          panel.spacing.y = unit(10, "pt"),
          legend.background = element_rect(fill = "transparent"))
}


cat(">>> Generating Visualizations...\n")

# Color palettes
my_colors <- c("Duration" = "#c0321a", "Formant" = "#547bb4", 
               "Voice" = "#629c35", "Coda" = "#F0E442", "Pitch" = '#6c61af', "Other" = "grey")

# Figure 1: SHAP Ranking Stability Bump Chart
label_data_bump <- stab_shap$Table_A %>% filter(Rep == 20)
p_bump <- ggplot(stab_shap$Table_A, aes(x = factor(Rep), y = Rank, group = Feature, color = Feature)) +
  geom_line(alpha = 0.5, linewidth = 1) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_text_repel(data = label_data_bump, aes(label = Feature), 
                  nudge_x = 0.5, direction = "y", hjust = 0, segment.color = "grey", size = 3.5, fontface = "bold") +
  scale_y_reverse(breaks = 1:22) + 
  scale_x_discrete(expand = expansion(mult = c(0.05, 0.2))) +
  labs(title = "SHAP Feature Ranking Stability across Repetitions", x = "Repetition (Random Seed)", y = "Rank (1 is Best)") +
  theme_classic(base_size = 14) + theme(legend.position = "none")

# Figure 2: Main Feature Importance Bar Plot + Loess Trends
p_main_imp <- ggplot() +
  geom_col(data = shap_imp_res$feat_imp, aes(x = importance, y = feature, fill = group), width = 0.9) +
  geom_errorbar(data = shap_imp_res$feat_imp, aes(y = feature, xmin = CI_Lower, xmax = CI_Upper), width = 0.3, color = "#333333", linewidth = 0.3) +
  geom_path(data = trend_df, aes(x = x_plot, y = y_plot, group = feature), color = "#444444", linewidth = 0.5) +
  scale_fill_manual(values = my_colors) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(x = "Mean |SHAP Value|", y = NULL) +
  theme_classic() + theme(legend.position = "none")
ggsave("Feature Importance.pdf", plot = p_main_imp, device = cairo_pdf, width = 3.5, height = 5)

# Figure 3: Sector Width Rose Diagram
df_odd <- rose_data[rose_data$ord %% 2 == 1, ]
p_rose <- ggplot(rose_data) +
  geom_rect(data = df_odd, aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = 0.05*ord), fill = '#f5f5f5', color = "white") +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = 0.05*ord, ymax = 0.1 + 0.05*ord, fill = group), color = "white") +
  coord_polar(theta = "x", start = 0, clip = "off") +  
  scale_fill_manual(values = my_colors) +
  scale_y_continuous(limits = c(0, 1.2), expand = c(0, 0)) +
  geom_text(aes(x = x_center, y = 0.2 + 0.05*ord, label = ifelse(importance/sum(importance) > 0.05, paste0(round(importance/sum(importance)*100, 0), "%"), "")), size = 3) +
  theme_void() + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0, "pt"))
ggsave("p_rose.pdf", plot = p_rose, device = cairo_pdf, width = 3.5, height = 5)

# Figure 4: Grouped Feature Importance
p_grp_imp <- ggplot(group_imp_df, aes(x = importance, y = group, fill = group)) +
  geom_col(width = 0.6) +
  scale_fill_manual(values = my_colors) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(x = "Feature Importance", y = NULL) +
  theme_classic() + theme(legend.position = "none")
ggsave("p_grp_imp.pdf", plot = p_grp_imp, device = cairo_pdf, width = 2.8, height = 2.5)

# Figure 5: Precision Review Distributions
p_pr_dist <- ggplot(pr_dist, aes(x = age, y = .estimate, fill = age)) +
  geom_violin(alpha = 0.5) + geom_jitter(width = 0.1, alpha = 0.5) +
  facet_grid(register ~ vowel) +
  labs(title = "PR-AUC Distribution Structure (20 Runs)", y = "PR-AUC Assessment", x = "Age Stratification Category") +
  theme_minimal()
print(p_pr_dist)

# Figure 6: Baseline CatBoost vs. Logistic Regression Precision
summary_stats_pr <- pr_dist %>%
  rename(pr_auc = .estimate) %>%
  group_by(age, vowel, register) %>%
  summarise(
    n = sum(!is.na(pr_auc)),            # 计算非 NA 的样本量
    mean_auc = mean(pr_auc, na.rm = TRUE), # 计算均值，忽略 NA
    sd_auc = sd(pr_auc, na.rm = TRUE),     # 计算标准差，忽略 NA
    .groups = 'drop'
  ) %>%
  mutate(formatted_auc = paste0(format(round(mean_auc, 3), nsmall = 3), 
                                " ± ", 
                                format(round(sd_auc, 3), nsmall = 3))) %>%
  select(age, vowel, register, formatted_auc) %>%
  pivot_wider(names_from = age, values_from = formatted_auc)

pr_labels <- summary_stats_pr %>%
  pivot_longer(cols = c(Old, Young), 
               names_to = "age", 
               values_to = "auc_text") %>%
  mutate(label = paste0(age, ": ", auc_text))

p_compare <- ggplot() +
  geom_ribbon(data = pr_compare, aes(x = recall, y = precision, color = age, linetype = Model,ymin = lower, ymax = upper, fill = age), alpha = 0.15, color = NA) +
  geom_line(data = pr_compare, aes(x = recall, y = precision, color = age, linetype = Model),linewidth = 0.5) +
  geom_text(data = pr_labels, 
            aes(x = 0.2, y = ifelse(age == "Old", 0.4, 0.3), label = auc_text, color = age), # 根据需要调整 x, y 位置
            size = 4, 
            show.legend = FALSE) + 
  facet_grid(register ~ vowel) +
  scale_linetype_manual(values = c("CatBoost" = 1, "LR (Baseline)" = 2)) +
  scale_color_brewer(palette = "Set1") + 
  scale_fill_brewer(palette = "Set1") +
  labs(x = "Recall", y = "Precision", linetype = "Models") + 
  theme_my_style() + 
  theme(legend.position = "right")
ggsave("p_compare.pdf", plot = p_compare, device = cairo_pdf, width = 15, height = 7)

p_age_diff <- ggplot() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_density_ridges(data = age_diffs,aes(x= diff ,y = .category), linewidth = NA, color = 'grey80',alpha=0.5,scale = 0.8,rel_min_height = 0.01) +
  geom_linerange(data = summary_diffs, aes(y = Feature, x = diff, linewidth = as.factor(.width), color = hdi0, xmin = .lower, xmax = .upper)) +
  geom_point(data = summary_diffs, aes(y = Feature, x = diff),size = 0.7, color = 'white') +
    facet_grid(register ~ vowel) +
  scale_color_manual(values = c('#fc8d59','grey25')) + 
  scale_linewidth_manual(values = c(1.5,1,0.5))+
  scale_x_continuous(breaks = seq(-0.1,0.1,0.05),expand = 0.01)+
  labs(x = 'Posterior Relative Weight Difference (Young - Old)', y = 'Feature',
       color = 'CrI includes 0',linewidth = 'CrI level')+
  theme_my_style()+
  theme(legend.position = "right", panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5))

combined_plot <- (p_compare / p_age_diff) + 
  plot_annotation(tag_levels = 'A')

ggsave("p_combined_plot.pdf", plot = combined_plot, device = cairo_pdf, width = 15, height = 10)

# Figure 7: Correlations between cue weights and birth year/dominance
p_cor = ggplot(cor_results, aes(x = feature, y = r, fill = target_var)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.6),width = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_text(
    aes(label = sig, y = r + ifelse(r >= 0, 0.07, -0.07)), 
    position = position_dodge(width = 0.6),
    vjust = 0.5,hjust=0.5
  ) +
  coord_flip()+
  scale_y_continuous(
    limits = c(-1.05, 1.05)
  )+
  facet_grid(register ~ vowel) +
  scale_fill_brewer(palette = "Set2", name = "Target Variable") +
  labs(
    x = "Feature",
    y = "Pearson Correlation (r)"
  ) +
  theme_my_style()+
  theme(legend.position = "right", panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5))

ggsave("p_cor.pdf", plot = p_cor, device = cairo_pdf, width = 15, height = 5)

# Clustering Visualization

# Best K Visualization
p.K = plot(2:6, cluster_res$nb_res$All.index, type="b", pch=19, frame=FALSE, 
     xlab="Number of clusters K", 
     ylab="Average Silhouette Width",
     main="Silhouette Method for Optimal K")

mat <- as.matrix(cluster_res$clustering_data)
col_fun = colorRamp2(c(min(mat), 0, max(mat)), c("navy", "white", "firebrick3"))
dend = as.dendrogram(cluster_res$hc)
clusters <- cutree(cluster_res$hc, k = 3)

p_heatmap = Heatmap(mat,
                    col = col_fun,
                    row_split = 3, 
                    cluster_rows = dend,           # 传入带颜色的树对象
                    row_dend_gp = gpar(lwd = 1),# 设置全局线条粗细
                    column_dend_gp = gpar(lwd = 1),
                    rect_gp = gpar(col = "white", lwd = 0.5),
                    row_dend_width = unit(4, "cm"),
                    column_dend_height = unit(2, "cm"),
                    heatmap_legend_param = list(title = "CLR_weight")
)

cairo_pdf("p_heatmap.pdf", width = 9, height = 8)
draw(p_heatmap)
dev.off()

# PCA Visualization
loadings <- as.data.frame(cluster_res$pca$rotation)
loadings$Variable <- rownames(loadings)
loadings$contribution <- sqrt(loadings$PC1^2 + loadings$PC2^2)
loadings_top <- loadings[order(-loadings$contribution), ][1:15, ]

eig_val <- get_eigenvalue(cluster_res$pca)
pc1_var <- eig_val$variance.percent[1]
pc2_var <- eig_val$variance.percent[2]

p_pca <- ggplot() +
  geom_point(data = cluster_res$df_pca, 
             aes(x = PC1, 
                 y = PC2, 
                 shape = as.factor(cluster), 
                 fill = YearOfBirth, 
                 size = dominance),
             color = 'white') +
  geom_text(data = cluster_res$df_pca,aes(x = PC1, 
                                          y = PC2,
                                          label=id))+
  scale_fill_distiller(palette = "Spectral", direction = 1)+
  scale_shape_manual(name = "Cluster", values = c(21, 22, 24))+
  scale_size_continuous(range = c(3, 8))+
  geom_segment(data = loadings_top, 
               aes(x = 0, y = 0, xend = PC1, yend = PC2),
               arrow = arrow(length = unit(0.2, "cm")), color = "black",alpha=0.5) +
  geom_text_repel(data = loadings_top,
                  aes(x = PC1, y = PC2, label = Variable),
                  color = "black", size = 3,alpha=0.5) +
  geom_segment(data = vec_df, aes(x = 0, y = 0, xend = PC1, yend = PC2, color=Variable),
               arrow = arrow(length = unit(0.2, "cm")), size = 1) +
  labs(x = sprintf("PC1 (%.2f%%)", pc1_var), y = sprintf("PC2 (%.2f%%)", pc2_var),
       shape = "Cluster", fill = "YearOfBirth", size = "Dominance")+
  guides(
    shape = guide_legend(override.aes = list(color = "black",fill = "black",size = 5)),
    size = guide_legend(override.aes = list(color = "black",fill = 'black'))
  )+
  theme_my_style()+theme(legend.position = "right", panel.grid = element_blank())

ggsave("p_pca.pdf", plot = p_pca, device = cairo_pdf, width = 6, height = 5)

