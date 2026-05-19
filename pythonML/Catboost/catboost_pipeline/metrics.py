import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold


def recover_catboost_metrics_with_ratio(X, y, save_dir, n_splits=10, n_repeats=20):
    """Recover fold-level F1, PR-AUC and positive ratio from saved CatBoost models."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    model_dir = os.path.join(save_dir, "models")
    results_list = []

    print(f"Recovering metrics from {n_splits * n_repeats} models...")

    for i, (_, test_idx) in enumerate(rskf.split(X, y)):
        repeat_num = i // n_splits + 1
        fold_num = i % n_splits + 1
        step_name = f"Rep{repeat_num}_Fold{fold_num}"

        model_filename = f"cb_model_{step_name}.joblib"
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            print(f"Warning: model file missing {model_path}, skipping")
            continue

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        pos_ratio = y_test.mean()

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        results_list.append(
            {
                "Repeat_ID": repeat_num,
                "Fold_ID": fold_num,
                "Model_Name": step_name,
                "F1_Score": f1,
                "PR_AUC": pr_auc,
                "Pos_Ratio": pos_ratio,
            }
        )

    return pd.DataFrame(results_list)


def recover_catboost_metrics_with_ratio_grouped(
    X, y, groups, save_dir, n_splits=10, n_repeats=20
):
    """Recover fold-level F1, PR-AUC and positive ratio from saved grouped CatBoost models."""
    model_dir = os.path.join(save_dir, "models")
    results_list = []

    print(f"Recovering metrics from {n_splits * n_repeats} models (Grouped)...")

    for repeat in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)

        for fold, (_, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
            repeat_num = repeat + 1
            fold_num = fold + 1
            step_name = f"Rep{repeat_num}_Fold{fold_num}"

            model_filename = f"cb_model_{step_name}.joblib"
            model_path = os.path.join(model_dir, model_filename)

            if not os.path.exists(model_path):
                print(f"Warning: model file missing {model_path}, skipping")
                continue

            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            pos_ratio = y_test.mean()

            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            results_list.append(
                {
                    "Repeat_ID": repeat_num,
                    "Fold_ID": fold_num,
                    "Model_Name": step_name,
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "Accuracy": accuracy,
                    "Recall": recall,
                    "Precision": precision,
                    "F1_Score": f1,
                    "ROC_AUC": roc_auc,
                    "PR_AUC": pr_auc,
                    "Pos_Ratio": pos_ratio,
                }
            )

    return pd.DataFrame(results_list)
