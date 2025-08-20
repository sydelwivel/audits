# backend/synthetic.py
"""
Synthetic data generator that preserves categorical columns (including sensitive attrs)
and balances the target using SMOTENC when available. Falls back to simple upsampling.
Functions:
  - generate_synthetic_data(df, target_column, random_state=42)
Returns:
  - synthetic_df (pandas.DataFrame)
Notes:
  - Does NOT use Streamlit internals (pure backend).
  - Requires scikit-learn; imblearn is optional but recommended.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np

# optional imports
try:
    from imblearn.over_sampling import SMOTENC
    _HAS_SMOTENC = True
except Exception:
    _HAS_SMOTENC = False

from sklearn.preprocessing import LabelEncoder

def _upsample_minority_simple(df: pd.DataFrame, target_column: str, random_state: int = 42) -> pd.DataFrame:
    """
    Simple fallback upsampling: replicate minority-class rows with replacement until classes are balanced.
    Keeps all columns as-is (categorical columns preserved).
    """
    rng = np.random.RandomState(random_state)
    counts = df[target_column].value_counts()
    if counts.empty:
        return df.copy()
    max_n = counts.max()
    parts = []
    for cls, n in counts.items():
        cls_df = df[df[target_column] == cls]
        if n < max_n:
            reps = max_n - n
            # sample with replacement
            sampled = cls_df.sample(reps, replace=True, random_state=rng)
            new_cls_df = pd.concat([cls_df, sampled], ignore_index=True)
        else:
            new_cls_df = cls_df.copy()
        parts.append(new_cls_df)
    res = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=rng).reset_index(drop=True)
    return res

def generate_synthetic_data(df: pd.DataFrame, target_column: str, random_state: int = 42) -> pd.DataFrame:
    """
    Generate balanced synthetic dataset while preserving categorical columns.

    Args:
      df: original DataFrame (must contain target_column)
      target_column: name of the target column to balance
      random_state: RNG seed

    Returns:
      synthetic_df: balanced DataFrame containing all original columns and same column order.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not in dataframe columns")

    df = df.copy().reset_index(drop=True)

    # drop rows with missing target to avoid issues
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    if df.shape[0] == 0:
        return df

    # detect categorical columns (object / category)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # but do not treat the target as a feature
    if target_column in categorical_cols:
        # keep it in cat list for correct handling but will be separated
        pass

    # Separate X and y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # If only one class, nothing to do
    if y.nunique() <= 1:
        return df.copy()

    # Strategy 1: use SMOTENC if available
    if _HAS_SMOTENC:
        try:
            # We'll encode categorical columns to integer labels (LabelEncoder) to use SMOTENC,
            # then inverse-transform after sampling.
            X_enc = X.copy()
            encoders = {}  # col -> LabelEncoder
            cat_idx_positions = []
            col_list = list(X_enc.columns)
            for i, col in enumerate(col_list):
                if col in categorical_cols:
                    le = LabelEncoder()
                    # fillna with sentinel string
                    X_enc[col] = X_enc[col].astype(str).fillna("##MISSING##")
                    X_enc[col] = le.fit_transform(X_enc[col])
                    encoders[col] = le
                    cat_idx_positions.append(i)
                else:
                    # numeric type: coerce to float (keeps NaNs)
                    X_enc[col] = pd.to_numeric(X_enc[col], errors='coerce').fillna(0.0)

            # SMOTENC expects numpy arrays
            X_vals = X_enc.values
            # y might be non-numeric (strings); convert using LabelEncoder too
            y_series = pd.Series(y).astype(str)
            y_le = LabelEncoder()
            y_enc = y_le.fit_transform(y_series)

            smote = SMOTENC(categorical_features=cat_idx_positions, random_state=random_state)
            X_res, y_res = smote.fit_resample(X_vals, y_enc)

            # recreate DataFrame
            X_res_df = pd.DataFrame(X_res, columns=col_list)

            # inverse-transform categorical columns
            for col in categorical_cols:
                if col in encoders:
                    le = encoders[col]
                    # values in X_res_df[col] are integers possibly outside training classes;
                    # clip to nearest known class index to avoid errors
                    vals = X_res_df[col].round().astype(int).values
                    # handle out-of-range by mapping via modulo or clipping
                    vals_clipped = np.clip(vals, vals.min(), vals.max())
                    try:
                        inv = le.inverse_transform(vals_clipped)
                    except Exception:
                        # fallback: map any unseen value to the first class
                        inv = []
                        classes = list(le.classes_)
                        for v in vals_clipped:
                            idx = int(v) if 0 <= int(v) < len(classes) else 0
                            inv.append(classes[idx])
                        inv = np.array(inv)
                    # restore np.nan marker if sentinel
                    inv = pd.Series(inv).replace("##MISSING##", np.nan)
                    X_res_df[col] = inv.values
                else:
                    # numeric column: keep as float; if originally int, cast to int
                    try:
                        orig_dtype = X[col].dtype
                        if np.issubdtype(orig_dtype, np.integer):
                            X_res_df[col] = X_res_df[col].round().astype(int)
                        else:
                            X_res_df[col] = pd.to_numeric(X_res_df[col], errors='coerce')
                    except Exception:
                        X_res_df[col] = pd.to_numeric(X_res_df[col], errors='coerce')

            # reconstruct y into original labels
            y_res_labels = y_le.inverse_transform(y_res)

            synthetic_df = X_res_df.copy()
            synthetic_df[target_column] = pd.Series(y_res_labels)

            # Ensure same column order as original
            for c in df.columns:
                if c not in synthetic_df.columns:
                    synthetic_df[c] = np.nan
            synthetic_df = synthetic_df[df.columns.tolist()]

            # Shuffle rows
            synthetic_df = synthetic_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            return synthetic_df

        except Exception as e:
            # fall through to simple upsampling fallback
            # print but do not raise
            print(f"[synthetic] SMOTENC failed, falling back to simple upsampling. Error: {e}")

    # Fallback: simple upsample minority classes (keeps categorical columns intact)
    try:
        return _upsample_minority_simple(df, target_column, random_state=random_state)
    except Exception as e:
        print(f"[synthetic] fallback upsample failed: {e}")
        # as a last resort, return original df
        return df.copy()
