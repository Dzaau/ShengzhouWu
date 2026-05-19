import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def _resolve_scoring(scoring):
    scoring = scoring.lower()
    if scoring in {"auprc", "pr_auc", "average_precision"}:
        return "auprc"
    if scoring in {"roc_auc", "auc"}:
        return "roc_auc"
    raise ValueError("scoring must be one of: 'auprc', 'pr_auc', 'average_precision', 'roc_auc'")


def _score_binary(y_true, y_proba, scoring):
    if scoring == "auprc":
        return average_precision_score(y_true, y_proba)
    return roc_auc_score(y_true, y_proba)


def tune_catboost_classifier(
    X,
    y,
    groups,
    cat_features,
    feature_names=None,
    test_size=0.2,
    random_state=42,
    stratify=True,
    n_trials_stage1=40,
    n_trials_stage2=20,
    n_splits=5,
    scoring="auprc",
    early_stopping_rounds=50,
    n_jobs=-1,
    plot_importance=True,
):
    """
    Two-stage Optuna tuning for CatBoost with fixed CV protocol.

    Stage 1 searches a broader parameter range, stage 2 narrows around stage-1 best values.
    """
    metric_name = _resolve_scoring(scoring)

    # Note: Use a GroupShuffleSplit if stratification is needed with groups,
    # or StratifiedGroupKFold to get a single train/test split.
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
    
    cv_outer = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_idx, test_idx = next(cv_outer.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx] if isinstance(groups, pd.Series) else groups[train_idx], groups.iloc[test_idx] if isinstance(groups, pd.Series) else groups[test_idx]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective_stage1(trial):
        imbalance_mode = trial.suggest_categorical("imbalance_mode", ["auto_balanced", "auto_sqrt_balanced", "none"])

        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": random_state,
            "thread_count": 1,
            "verbose": False,
        }

        if imbalance_mode == "auto_balanced":
            params["auto_class_weights"] = "Balanced"
        elif imbalance_mode == "auto_sqrt_balanced":
            params["auto_class_weights"] = "SqrtBalanced"

        fold_scores = []
        for train_idx_inner, valid_idx_inner in cv.split(X_train, y_train, groups=groups_train):
            X_tr = X_train.iloc[train_idx_inner]
            y_tr = y_train.iloc[train_idx_inner]
            X_val = X_train.iloc[valid_idx_inner]
            y_val = y_train.iloc[valid_idx_inner]

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=early_stopping_rounds,
            )
            y_val_proba = model.predict_proba(X_val)[:, 1]
            fold_scores.append(_score_binary(y_val, y_val_proba, metric_name))

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study_stage1 = optuna.create_study(direction="maximize", sampler=sampler)
    study_stage1.optimize(objective_stage1, n_trials=n_trials_stage1, n_jobs=n_jobs)

    best1 = study_stage1.best_params

    def objective_stage2(trial):
        iterations_low = max(200, int(best1["iterations"] * 0.7))
        iterations_high = min(1000, int(best1["iterations"] * 1.3))
        lr_low = max(0.005, best1["learning_rate"] * 0.6)
        lr_high = min(0.3, best1["learning_rate"] * 1.4)
        l2_low = max(0.5, best1["l2_leaf_reg"] * 0.5)
        l2_high = min(100.0, best1["l2_leaf_reg"] * 2.0)

        imbalance_mode = trial.suggest_categorical(
            "imbalance_mode",
            [best1.get("imbalance_mode", "auto_balanced"), "auto_balanced", "auto_sqrt_balanced", "none"],
        )

        params = {
            "iterations": trial.suggest_int("iterations", iterations_low, iterations_high),
            "learning_rate": trial.suggest_float("learning_rate", lr_low, lr_high, log=True),
            "depth": trial.suggest_int("depth", max(3, best1["depth"] - 2), min(7, best1["depth"] + 2)),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", l2_low, l2_high, log=True),
            "subsample": trial.suggest_float("subsample", max(0.5, best1["subsample"] - 0.2), min(1.0, best1["subsample"] + 0.2)),
            "rsm": trial.suggest_float("rsm", max(0.4, best1["rsm"] - 0.2), min(1.0, best1["rsm"] + 0.2)),
            "random_strength": trial.suggest_float(
                "random_strength",
                max(0.0, best1["random_strength"] - 2.0),
                min(10.0, best1["random_strength"] + 2.0),
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                max(0.0, best1["bagging_temperature"] - 3.0),
                min(20.0, best1["bagging_temperature"] + 3.0),
            ),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": random_state,
            "thread_count": n_jobs,
            "verbose": False,
        }

        if imbalance_mode == "auto_balanced":
            params["auto_class_weights"] = "Balanced"
        elif imbalance_mode == "auto_sqrt_balanced":
            params["auto_class_weights"] = "SqrtBalanced"

        fold_scores = []
        for train_idx_inner, valid_idx_inner in cv.split(X_train, y_train, groups=groups_train):
            X_tr = X_train.iloc[train_idx_inner]
            y_tr = y_train.iloc[train_idx_inner]
            X_val = X_train.iloc[valid_idx_inner]
            y_val = y_train.iloc[valid_idx_inner]

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=early_stopping_rounds,
            )
            y_val_proba = model.predict_proba(X_val)[:, 1]
            fold_scores.append(_score_binary(y_val, y_val_proba, metric_name))

        return float(np.mean(fold_scores))

    study_stage2 = optuna.create_study(direction="maximize", sampler=sampler)
    study_stage2.enqueue_trial({
        "iterations": best1["iterations"],
        "learning_rate": best1["learning_rate"],
        "depth": best1["depth"],
        "l2_leaf_reg": best1["l2_leaf_reg"],
        "subsample": best1["subsample"],
        "rsm": best1["rsm"],
        "random_strength": best1["random_strength"],
        "bagging_temperature": best1["bagging_temperature"],
        "imbalance_mode": best1.get("imbalance_mode", "auto_balanced"),
    })
    study_stage2.optimize(objective_stage2, n_trials=n_trials_stage2,n_jobs=n_jobs)

    best = study_stage2.best_params
    best_params = {
        "iterations": int(best["iterations"]),
        "learning_rate": float(best["learning_rate"]),
        "depth": int(best["depth"]),
        "l2_leaf_reg": float(best["l2_leaf_reg"]),
        "subsample": float(best["subsample"]),
        "rsm": float(best["rsm"]),
        "random_strength": float(best["random_strength"]),
        "bagging_temperature": float(best["bagging_temperature"]),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": random_state,
        "thread_count": n_jobs,
        "verbose": False,
    }

    imbalance_mode = best.get("imbalance_mode", "auto_balanced")
    if imbalance_mode == "auto_balanced":
        best_params["auto_class_weights"] = "Balanced"
    elif imbalance_mode == "auto_sqrt_balanced":
        best_params["auto_class_weights"] = "SqrtBalanced"

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
    )

    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_auprc = average_precision_score(y_test, y_pred_proba)
    report_text = classification_report(y_test, y_pred)

    if feature_names is None:
        feature_names = list(X.columns)

    importances = final_model.get_feature_importance()
    feat_imp_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    if plot_importance:
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(25), palette="viridis")
        plt.title("CatBoost Feature Importance (Final Model)", fontsize=15)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()
        plt.show()

    return {
        "best_params": best_params,
        "best_cv_score": float(study_stage2.best_value),
        "test_accuracy": float(test_accuracy),
        "test_auc": float(test_auc),
        "test_auprc": float(test_auprc),
        "classification_report": report_text,
        "feature_importance_df": feat_imp_df,
        "final_model": final_model,
        "optuna_study_stage1": study_stage1,
        "optuna_study_stage2": study_stage2,
        "scoring_used": metric_name,
    }
