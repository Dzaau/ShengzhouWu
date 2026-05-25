# CatBoost 2026 Pipeline

This folder provides an independent CatBoost implementation used in the study of checkedness in Shengzhou Wu Chinese:

- data preparation
- hyperparameter tuning
- repeated CV experiment
- SHAP + SHAP interaction export
- fold-level metrics recovery
- R-ready long-format exports

## Directory Layout

```text
pythonML/
  README.md
  Catboost/
    requirements.txt
    CatBoost_workflow.py
    catboost_pipeline/
      __init__.py
      data_prep.py
      tuning.py
      experiment.py
      metrics.py
      reporting.py
  /catboost_grouped_20260409
```

## Environment Setup

From the `pythonML/Catboost` folder:

```bash
pip install -r requirements.txt
```

## Main Workflow

Run `CatBoost_workflow.py` top-to-bottom (or execute cells with `#%%` blocks):

1. Load source data from `../backup_source.joblib`
2. Build native categorical features (`cat_features`) for CatBoost
3. Tune CatBoost with Optuna (two-stage search)
4. Run full repeated CV experiment (`5 folds x 20 repeats`)
5. Export:
   - `df_performance.csv`
   - `df_performance_all.csv`
   - `df_raw_shap.csv`
   - `full_shap_interaction_features.csv` (not analyzed in this study)

## Notes on Imbalanced Target

Default strategy uses CatBoost built-in class balancing:

- `auto_class_weights = "Balanced"` (or `"SqrtBalanced"` via tuning)

Recommended reporting metrics:

- PR-AUC
- F1
- Recall
- Precision
