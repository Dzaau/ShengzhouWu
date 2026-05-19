import joblib
import numpy as np
import pandas as pd


def load_original_data(joblib_path="backup_df_original20260124.joblib"):
    """Load the original dataframe from a joblib file."""
    df_original = joblib.load(joblib_path)
    print("done")
    return df_original


def prepare_features_and_target_catboost(df_original, include_vars=None):
    """
    Prepare native CatBoost inputs without one-hot encoding.

    Returns
    -------
    data : pd.DataFrame
        Filtered dataframe with synthetic ref column.
    X : pd.DataFrame
        Feature dataframe (native categorical columns kept as category dtype).
    y : pd.Series
        Target column.
    cat_features : list[str]
        Categorical feature names for CatBoost.
    """
    df = df_original.copy()

    var_to_select = [
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

    if include_vars is not None:
        include_vars = list(include_vars)
        missing_cols = [col for col in include_vars if col not in df.columns]
        if missing_cols:
            raise ValueError(f"include_vars has columns missing in df_original: {missing_cols}")
        if "MC_Ru" not in include_vars:
            include_vars.append("MC_Ru")
        var_to_select = include_vars

    data = df.loc[:, df.columns.isin(var_to_select)].copy()

    np.random.seed(42)
    data["ref"] = np.random.randn(len(data))

    y = data["MC_Ru"].copy()
    X = data.drop(columns=["MC_Ru"]).copy()

    cat_candidates = ["sex", "creak", "glottalStop", "condition", "edu"]
    cat_features = [col for col in cat_candidates if col in X.columns]

    for col in cat_features:
        X[col] = X[col].astype("category")

    print("done")
    return data, X, y, cat_features
