

#%%
import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.dirname(CURRENT_DIR)
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from Catboost.catboost_pipeline import (
    prepare_features_and_target_catboost,
    recover_catboost_metrics_with_ratio_grouped,
    run_catboost_shap_grouped,
    tune_catboost_classifier,
)
from rf_pipeline.interactions import build_full_shap_interaction_dataframe

#%%

#df_original = pd.read_excel(r"...\R code Final 202511\TheLastChance\RF_20260124.xlsx")
#joblib.dump(df_original, 'backup_df_original20260424RIGHT.joblib', compress=3)
#print("Loaded original Excel and saved as joblib for faster loading in the future.")


#%%
# 1) Load source data
source_joblib = os.path.join(CURRENT_DIR, "backup_source.joblib")
df_original = joblib.load(source_joblib)

include_vars = [
    "YearOfBirth",
    "sex",
    "edu",
    "dominance",
    "condition",
    "MC_Ru",
    "glottalStop",
    "pitch_height",
    "pitch_change",
    "creak",
    "CQ_z",
    "SQ_z",
    "voice_PC1",
    "voice_PC2",
    "voice_PC4",
    "voice_PC6",
    "voice_PC7",
    "voice_PC8",
    "voice_PC9",
    "duration_z",
    "F1_z",
    "F2_z",
]

data, X, y, cat_features = prepare_features_and_target_catboost(df_original, include_vars=include_vars)
subject_ids = df_original["id"].copy()
print(X.shape, y.shape)
print("Categorical features:", cat_features)


#%%
# 2) Tune CatBoost with Optuna
tuning_result = tune_catboost_classifier(
    X,
    y,
    subject_ids,
    cat_features=cat_features,
    feature_names=list(X.columns),
    n_trials_stage1=40,
    n_trials_stage2=20,
    n_splits=5,
    scoring="auprc",
    random_state=42,
    plot_importance=True,
    n_jobs=8
)

print("Best params:", tuning_result["best_params"])
print("Best CV score:", tuning_result["best_cv_score"])
print("Test AUC:", tuning_result["test_auc"])
print("Test AUPRC:", tuning_result["test_auprc"])

today = datetime.now().strftime("%Y%m%d")
save_dir = os.path.join(TEST_ROOT, f"catboost_grouped_{today}")
os.makedirs(save_dir, exist_ok=True)

best_params_path = os.path.join(save_dir, "cb_best_params.json")
with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(tuning_result["best_params"], f, ensure_ascii=False, indent=2)
print("Saved best params to:", best_params_path)


#%%
# 3) Run full CatBoost experiment (SHAP + interaction)
#today = datetime.now().strftime("%Y%m%d")
#save_dir = os.path.join(TEST_ROOT, f"catboost_grouped_{today}")
best_params_path = os.path.join(save_dir, "cb_best_params.json")
if os.path.exists(best_params_path):
    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
else:
    best_params = None

results = run_catboost_shap_grouped(
    X,
    y,
    subject_ids,
    df_original[["fileTag"]],
    cat_features,
    mode="both",
    model_params=best_params,
    n_splits=5,
    n_repeats=20,
    output_html="catboost_report_grouped.html",
    save_dir=save_dir,
    backup_model=True,
    backup_shap=True,
)

backup_result_path = os.path.join(save_dir, f"backup_result_{today}.joblib")
joblib.dump(results, backup_result_path, compress=3)
print("Saved backup result to:", backup_result_path)


#%%
# 4) Build full interaction dataframe with grouping
my_groups = {
    "Duration": ["duration_z"],
    "Coda": ["glottalStop"],
    "Voice": [
        "SQ_z",
        "CQ_z",
        "voice_PC1",
        "voice_PC2",
        "voice_PC4",
        "voice_PC6",
        "voice_PC7",
        "voice_PC8",
        "voice_PC9",
        "creak",
    ],
    "Formant": ["F1_z", "F2_z"],
    "Pitch": ["pitch_height", "pitch_change"],
    "YoB": ["YearOfBirth"],
    "dom": ["dominance"],
    "edu": ["edu"],
    "ref": ["ref"],
}

full_df = build_full_shap_interaction_dataframe(
    results_df=results,
    group_defs=my_groups,
    output_path=os.path.join(save_dir, "full_shap_interaction_features.csv"),
)
print("Full interaction dataframe shape:", full_df.shape)


#%%
# save_dir = os.path.join(TEST_ROOT, f"catboost_grouped_20260424")

# 5) Export for R analysis
perf_df = recover_catboost_metrics_with_ratio_grouped(
    X, y, subject_ids, save_dir=save_dir, n_splits=5, n_repeats=20
)
perf_df.to_csv(os.path.join(save_dir, "df_performance_all.csv"), index=False)

#%%
meta_cols = [
    "Repeat_ID",
    "Fold_ID",
    "fileTag",
    "Pred_Proba",
    "Pred_Cat",
    "Base_Value",
    "Model_Path",
    "Shap_Path",
    "Shap_Interaction_Path",
]
feature_cols = [c for c in results.columns if c not in meta_cols]

raw_shap_long = (
    results.reset_index()
    .melt(
        id_vars=["index", "Repeat_ID", "Fold_ID"],
        value_vars=feature_cols,
        var_name="Feature",
        value_name="SHAP_Value",
    )
    .rename(columns={"index": "Sample_ID", "Repeat_ID": "Rep", "Fold_ID": "Fold"})
)

raw_shap_long.to_csv(os.path.join(save_dir, "df_raw_shap.csv"), index=False)
print("Exported df_performance.csv and df_raw_shap.csv")

#%%
