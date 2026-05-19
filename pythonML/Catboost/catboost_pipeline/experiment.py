import os

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from .reporting import append_eval_section, build_html_header, close_html


def _extract_positive_class_values(shap_vals):
    if isinstance(shap_vals, list):
        return shap_vals[1]
    if len(shap_vals.shape) == 3:
        return shap_vals[:, :, 1]
    return shap_vals


def _extract_interaction_class1(shap_interactions_raw):
    if isinstance(shap_interactions_raw, list):
        return shap_interactions_raw[1]
    if len(shap_interactions_raw.shape) == 4:
        return shap_interactions_raw[:, :, :, 1]
    return shap_interactions_raw


def _extract_base_value(explainer):
    raw_base = explainer.expected_value
    if hasattr(raw_base, "__len__") and len(raw_base) > 1:
        base_val = raw_base[1]
    else:
        base_val = raw_base
    if hasattr(base_val, "item"):
        base_val = base_val.item()
    return base_val


def _get_catboost_model(random_seed, model_params=None):
    model_params = model_params or {}

    params = {
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": random_seed,
        "verbose": False,
    }

    params.update(model_params)
    params.setdefault("random_seed", random_seed)
    params.setdefault("verbose", False)
    return CatBoostClassifier(**params)


def _prepare_output_paths(mode, output_html, save_dir, backup_model=False, backup_shap=False):
    if mode in {"interaction", "both"}:
        os.makedirs(save_dir, exist_ok=True)
        model_dir = os.path.join(save_dir, "models")
        interaction_dir = os.path.join(save_dir, "shap_interactions")
        shap_values_dir = os.path.join(save_dir, "shap_values") if mode == "both" else None
        report_path = os.path.join(save_dir, output_html)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(interaction_dir, exist_ok=True)
        if shap_values_dir is not None:
            os.makedirs(shap_values_dir, exist_ok=True)
        return report_path, model_dir, shap_values_dir, interaction_dir

    if mode == "shap" and (backup_model or backup_shap):
        os.makedirs(save_dir, exist_ok=True)
        model_dir = os.path.join(save_dir, "models") if backup_model else None
        shap_values_dir = os.path.join(save_dir, "shap_values") if backup_shap else None
        report_path = os.path.join(save_dir, output_html)
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
        if shap_values_dir is not None:
            os.makedirs(shap_values_dir, exist_ok=True)
        return report_path, model_dir, shap_values_dir, None

    return output_html, None, None, None


def run_catboost_shap(
    X,
    y,
    data_meta,
    cat_features,
    mode="shap",
    model_params=None,
    n_splits=10,
    n_repeats=20,
    output_html="catboost_report.html",
    save_dir="catboost_results",
    backup_model=True,
    backup_shap=True,
):
    """Unified CatBoost workflow for SHAP, interaction SHAP, or both."""
    if mode not in {"shap", "interaction", "both"}:
        raise ValueError("mode must be one of 'shap', 'interaction', or 'both'")

    if mode in {"interaction", "both"}:
        backup_model = True
        backup_shap = True

    report_path, model_dir, shap_values_dir, interaction_dir = _prepare_output_paths(
        mode,
        output_html,
        save_dir,
        backup_model=backup_model,
        backup_shap=backup_shap,
    )

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    all_results = []
    html_content = build_html_header("CatBoost Evaluation Report")

    for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        repeat_num = i // n_splits + 1
        fold_num = i % n_splits + 1
        step_name = f"Rep{repeat_num}_Fold{fold_num}"

        print(f"Processing: {step_name} ...")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        cb_model = _get_catboost_model(42 + i, model_params=model_params)
        cb_model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test),
            use_best_model=True,
            early_stopping_rounds=50,
        )

        y_pred = cb_model.predict(X_test)
        y_pred_proba = cb_model.predict_proba(X_test)[:, 1]

        explainer = shap.TreeExplainer(cb_model)
        base_val = _extract_base_value(explainer)

        if mode == "interaction":
            model_filename = f"cb_model_{step_name}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(cb_model, model_path)

            shap_interactions_raw = explainer.shap_interaction_values(X_test)
            shap_interactions_class1 = _extract_interaction_class1(shap_interactions_raw)

            shap_filename = f"cb_shap_interaction_{step_name}.npz"
            shap_path = os.path.join(interaction_dir, shap_filename)
            np.savez_compressed(
                shap_path,
                interaction_values=shap_interactions_class1,
                feature_names=X.columns.values,
                sample_indices=X_test.index.values,
            )

            fold_df = pd.DataFrame(index=X_test.index)
            fold_df["Repeat_ID"] = repeat_num
            fold_df["Fold_ID"] = fold_num
            fold_df["Base_Value"] = base_val
            fold_df["Pred_Proba"] = y_pred_proba
            fold_df["Pred_Cat"] = y_pred
            fold_df["Model_Path"] = model_path
            fold_df["Shap_Interaction_Path"] = shap_path

            section_title = step_name
            html_content = append_eval_section(html_content, section_title, y_test, y_pred, y_pred_proba)
            all_results.append(fold_df)
            continue

        shap_vals = explainer.shap_values(X_test, check_additivity=False)
        shap_vals_pos = _extract_positive_class_values(shap_vals)

        fold_df = pd.DataFrame(shap_vals_pos, columns=X.columns, index=X_test.index)
        fold_df["Repeat_ID"] = repeat_num
        fold_df["Fold_ID"] = fold_num
        fold_df["Base_Value"] = base_val
        fold_df["Pred_Proba"] = y_pred_proba
        fold_df["Pred_Cat"] = y_pred

        if backup_model:
            model_filename = f"cb_model_{step_name}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(cb_model, model_path)
            fold_df["Model_Path"] = model_path

        if backup_shap:
            shap_filename = f"cb_shap_values_{step_name}.npz"
            shap_path = os.path.join(shap_values_dir, shap_filename)
            np.savez_compressed(
                shap_path,
                shap_values=shap_vals_pos,
                feature_names=X.columns.values,
                sample_indices=X_test.index.values,
            )
            fold_df["Shap_Path"] = shap_path

        if mode == "both":
            shap_interactions_raw = explainer.shap_interaction_values(X_test)
            shap_interactions_class1 = _extract_interaction_class1(shap_interactions_raw)
            interaction_filename = f"cb_shap_interaction_{step_name}.npz"
            interaction_path = os.path.join(interaction_dir, interaction_filename)
            np.savez_compressed(
                interaction_path,
                interaction_values=shap_interactions_class1,
                feature_names=X.columns.values,
                sample_indices=X_test.index.values,
            )
            fold_df["Shap_Interaction_Path"] = interaction_path

        section_title = step_name
        html_content = append_eval_section(html_content, section_title, y_test, y_pred, y_pred_proba)
        all_results.append(fold_df)

    html_content = close_html(html_content)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report saved to: {report_path}")

    results_df = pd.concat(all_results).sort_index()
    final_export = results_df.join(data_meta, how="left")
    return final_export


def load_interaction_matrix(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data["interaction_values"], data["feature_names"], data["sample_indices"]


def load_shap_values(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data["shap_values"], data["feature_names"], data["sample_indices"]


def run_catboost_shap_grouped(
    X,
    y,
    groups,
    data_meta,
    cat_features,
    mode="shap",
    model_params=None,
    n_splits=10,
    n_repeats=20,
    output_html="catboost_report.html",
    save_dir="catboost_results",
    backup_model=True,
    backup_shap=True,
):
    """Unified CatBoost workflow for SHAP, interaction SHAP, or both, using repeated StratifiedGroupKFold."""
    if mode not in {"shap", "interaction", "both"}:
        raise ValueError("mode must be one of 'shap', 'interaction', or 'both'")

    if mode in {"interaction", "both"}:
        backup_model = True
        backup_shap = True

    report_path, model_dir, shap_values_dir, interaction_dir = _prepare_output_paths(
        mode,
        output_html,
        save_dir,
        backup_model=backup_model,
        backup_shap=backup_shap,
    )

    all_results = []
    html_content = build_html_header("CatBoost Evaluation Report (Grouped)")

    for repeat in range(n_repeats):
        # We manually iterate for n_repeats and use different seeds for the StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
            repeat_num = repeat + 1
            fold_num = fold + 1
            step_name = f"Rep{repeat_num}_Fold{fold_num}"

            print(f"Processing: {step_name} ...")

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            cb_model = _get_catboost_model(42 + repeat * n_splits + fold, model_params=model_params)
            cb_model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                eval_set=(X_test, y_test),
                use_best_model=True,
                early_stopping_rounds=50,
            )

            y_pred = cb_model.predict(X_test)
            y_pred_proba = cb_model.predict_proba(X_test)[:, 1]

            explainer = shap.TreeExplainer(cb_model)
            base_val = _extract_base_value(explainer)

            if mode == "interaction":
                # Save Model
                model_filename = f"cb_model_{step_name}.joblib"
                model_path = os.path.join(model_dir, model_filename)
                joblib.dump(cb_model, model_path)

                # Interactions
                shap_interactions_raw = explainer.shap_interaction_values(X_test)
                shap_interactions_class1 = _extract_interaction_class1(shap_interactions_raw)

                shap_filename = f"cb_shap_interaction_{step_name}.npz"
                shap_path = os.path.join(interaction_dir, shap_filename)
                np.savez_compressed(
                    shap_path,
                    interaction_values=shap_interactions_class1,
                    feature_names=X.columns.values,
                    sample_indices=X_test.index.values,
                )

                fold_df = pd.DataFrame(index=X_test.index)
                fold_df["Repeat_ID"] = repeat_num
                fold_df["Fold_ID"] = fold_num
                fold_df["Base_Value"] = base_val
                fold_df["Pred_Proba"] = y_pred_proba
                fold_df["Pred_Cat"] = y_pred
                fold_df["Model_Path"] = model_path
                fold_df["Shap_Interaction_Path"] = shap_path

                section_title = step_name
                html_content = append_eval_section(html_content, section_title, y_test, y_pred, y_pred_proba)
                all_results.append(fold_df)
                continue

            # Standard SHAP Mode
            shap_vals = explainer.shap_values(X_test, check_additivity=False)
            shap_vals_pos = _extract_positive_class_values(shap_vals)

            fold_df = pd.DataFrame(shap_vals_pos, columns=X.columns, index=X_test.index)
            fold_df["Repeat_ID"] = repeat_num
            fold_df["Fold_ID"] = fold_num
            fold_df["Base_Value"] = base_val
            fold_df["Pred_Proba"] = y_pred_proba
            fold_df["Pred_Cat"] = y_pred

            if backup_model:
                model_filename = f"cb_model_{step_name}.joblib"
                model_path = os.path.join(model_dir, model_filename)
                joblib.dump(cb_model, model_path)
                fold_df["Model_Path"] = model_path

            if backup_shap:
                shap_filename = f"cb_shap_values_{step_name}.npz"
                shap_path = os.path.join(shap_values_dir, shap_filename)
                np.savez_compressed(
                    shap_path,
                    shap_values=shap_vals_pos,
                    feature_names=X.columns.values,
                    sample_indices=X_test.index.values,
                )
                fold_df["Shap_Path"] = shap_path

            if mode == "both":
                shap_interactions_raw = explainer.shap_interaction_values(X_test)
                shap_interactions_class1 = _extract_interaction_class1(shap_interactions_raw)
                interaction_filename = f"cb_shap_interaction_{step_name}.npz"
                interaction_path = os.path.join(interaction_dir, interaction_filename)
                np.savez_compressed(
                    interaction_path,
                    interaction_values=shap_interactions_class1,
                    feature_names=X.columns.values,
                    sample_indices=X_test.index.values,
                )
                fold_df["Shap_Interaction_Path"] = interaction_path

            section_title = step_name
            html_content = append_eval_section(html_content, section_title, y_test, y_pred, y_pred_proba)
            all_results.append(fold_df)

    html_content = close_html(html_content)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report saved to: {report_path}")

    results_df = pd.concat(all_results).sort_index()
    final_export = results_df.join(data_meta, how="left")
    return final_export
