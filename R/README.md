## Overview
This project contains the modularized codebase for the CatBoost analytical pipeline. Originally, data preprocessing, statistical inference logic, and visualization were tightly coupled in linear scripts. This repository cleanly abstracts function definitions by domains into individual `0X_...R` files, standardizing inputs and outputs as data frames.

The executable entry point is `main_CatBoost.R`, which acts as the orchestrator. It suppresses warnings, pulls required models, processes statistical significance with bootstrapped CI bands, conducts stability checking (Kendall's W / Friedman test), executes PR-AUC interpolation, generates bayesian Dirichlet model groupings via `brms`, and constructs individual-level analysis.

### Execution Guide
1. Ensure your current R session directory is set into the `CatBoostR_Cleaned` folder.
2. Verify that generic variables (such as `data.for.ML` or output variables from previous runs mapping your IDs) are pre-loaded in the global environment, OR adjust standard `load_base_data()` triggers in `01_data_prep.R` if calling isolated arrays.
3. Check and adjust the `data_workspace` and `interactions_path` path definitions at the top of `main_CatBoost.R`.
4. Ensure the `.RData` bayesian pre-trained datasets are in this folder unless you intend to train anew `(force_retrain = TRUE)`.
5. Run the master script: `source("main_CatBoost.R")`

### Modules
- **`01_data_prep.R`**: Extracts raw SHAP and test metric data structures from CSVs, resolving pivot structures into merged tables with primary mapping frames.
- **`02_logistic_baseline.R`**: Houses isolated generic `glm` algorithms serving as comparative baseline precision-recall validations.
- **`03_stability.R`**: Compiles ranking stabilities mapping cross-repetition Kendall variance and non-parametric evaluations.
- **`04_overall_shap.R`**: Manages mean extraction with subset Loess spline interpolation tracks supporting Rose and Bar importance charting parameters calculation.
- **`05_prauc.R`**: Groups validation mapping Precision-Recall parameters formatting standardized uniform scales.
- **`06_bayesian_models.R`**: Abstracts Stan compilation configuration structures and agglomerated CLR transformation mappings.