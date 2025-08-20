# backend/mitigation.py
"""
Bias Mitigation Utilities (no external heavy deps)

Implements the classic Reweighing technique (Kamiran & Calders):
  w(s, y) = P(S=s) * P(Y=y) / P(S=s, Y=y)
Weights are normalized to mean 1.0 for numerical stability.

Provides:
- reweighing_weights(df, sensitive_col, target_col) -> pd.Series (aligned to df.index)
- resample_by_weights(df, weights, n_samples=None, random_state=42) -> pd.DataFrame
- debias_dataset(df, sensitive_col, target_col, strategy="reweigh_resample", random_state=42)
    -> (debiased_df, weights, report_dict)

- Optional: decorrelate_numeric_features(df, sensitive_col)
  Reduces linear correlation between numeric features and the sensitive attribute
  by removing OLS-fitted component per feature.

The main aim is *dataset-level* mitigation. For model training, pass the weights
to algorithms that support sample_weight.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

def _prob(series) -> pd.Series:
    """Return empirical probability of each value in a Series."""
    vc = series.value_counts(dropna=False)
    total = vc.sum()
    return (vc / max(total, 1.0))

def reweighing_weights(df: pd.DataFrame, sensitive_col: str, target_col: str) -> pd.Series:
    """
    Compute reweighing weights per row: w(s,y) = P(S=s)P(Y=y) / P(S=s,Y=y)
    Returns: pd.Series aligned to df.index (NaN rows get weight=1.0)
    """
    if sensitive_col not in df.columns or target_col not in df.columns:
        raise ValueError("sensitive_col/target_col not found in dataframe")

    s = df[sensitive_col].astype(str)
    y = df[target_col].astype(str)

    # probabilities
    p_s = _prob(s)
    p_y = _prob(y)
    p_sy = _prob(pd.Series(list(zip(s, y))))

    # compute weights per (s,y)
    weights = []
    for si, yi in zip(s, y):
        key = (si, yi)
        ps = p_s.get(si, np.nan)
        py = p_y.get(yi, np.nan)
        psy = p_sy.get(key, np.nan)
        if pd.isna(ps) or pd.isna(py) or pd.isna(psy) or psy == 0:
            w = 1.0
        else:
            w = (ps * py) / psy
        weights.append(float(w))

    w_series = pd.Series(weights, index=df.index).fillna(1.0)

    # normalize weights to mean 1 to keep magnitudes reasonable
    mean_w = float(w_series.mean()) if len(w_series) else 1.0
    if mean_w > 0:
        w_series = w_series / mean_w
    return w_series

def resample_by_weights(
    df: pd.DataFrame,
    weights: pd.Series,
    n_samples: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Weighted resampling with replacement to create a 'debiased' dataset.
    - n_samples: if None, keep same size as original.
    """
    if n_samples is None:
        n_samples = len(df)
    weights = pd.Series(weights).reindex(df.index).fillna(1.0).astype(float)
    # clip tiny negatives / zeros for safety
    weights = weights.clip(lower=1e-9)
    probs = weights / weights.sum()
    rng = np.random.RandomState(random_state)
    idx = rng.choice(df.index.values, size=n_samples, replace=True, p=probs.values)
    return df.loc[idx].reset_index(drop=True)

def label_parity_diff(df: pd.DataFrame, sensitive_col: str, target_col: str) -> float:
    """
    Dataset-level 'Demographic Parity' on labels:
    max group positive-rate minus min group positive-rate (binary or multi-label treated as 1-vs-rest
    by choosing the mode as 'positive').
    """
    y = df[target_col]
    s = df[sensitive_col].astype(str)
    # binarize y: choose most frequent value as positive, rest as 0
    if y.nunique(dropna=True) == 0:
        return 0.0
    mode_val = y.mode(dropna=True).iloc[0]
    yb = (y == mode_val).astype(int)
    rates = yb.groupby(s).mean().dropna()
    if len(rates) <= 1:
        return 0.0
    return float(rates.max() - rates.min())

def debias_dataset(
    df: pd.DataFrame,
    sensitive_col: str,
    target_col: str,
    strategy: str = "reweigh_resample",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Main entry:
      strategy:
        - "reweigh_resample": compute weights, then weighted-resample dataset
        - "reweigh_only": compute weights only (return same df), for use in model training
        - "decorrelate_numeric": decorrelate numeric features from sensitive attribute

    Returns:
      debiased_df, weights_series, report_dict
    """
    if sensitive_col not in df.columns or target_col not in df.columns:
        raise ValueError("sensitive_col/target_col not found in dataframe")

    before_lpd = label_parity_diff(df, sensitive_col, target_col)

    if strategy == "reweigh_resample":
        w = reweighing_weights(df, sensitive_col, target_col)
        debiased = resample_by_weights(df, w, n_samples=len(df), random_state=random_state)
        after_lpd = label_parity_diff(debiased, sensitive_col, target_col)
        report = {
            "strategy": strategy,
            "before_label_parity_diff": round(before_lpd, 6),
            "after_label_parity_diff": round(after_lpd, 6),
            "note": "Dataset resampled with reweighing to reduce label distribution gaps across groups."
        }
        return debiased, w.reindex(df.index), report

    elif strategy == "reweigh_only":
        w = reweighing_weights(df, sensitive_col, target_col)
        after_lpd = before_lpd  # data unchanged
        report = {
            "strategy": strategy,
            "before_label_parity_diff": round(before_lpd, 6),
            "after_label_parity_diff": round(after_lpd, 6),
            "note": "Weights computed; pass as sample_weight during training."
        }
        return df.copy(), w, report

    elif strategy == "decorrelate_numeric":
        debiased = decorrelate_numeric_features(df, sensitive_col)
        # weights not used here
        w = pd.Series(1.0, index=df.index)
        after_lpd = label_parity_diff(debiased, sensitive_col, target_col)
        report = {
            "strategy": strategy,
            "before_label_parity_diff": round(before_lpd, 6),
            "after_label_parity_diff": round(after_lpd, 6),
            "note": "Numeric features were residualized on the sensitive attribute."
        }
        return debiased, w, report

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def decorrelate_numeric_features(df: pd.DataFrame, sensitive_col: str) -> pd.DataFrame:
    """
    Remove linear component of numeric features explained by sensitive attribute (one-hot encoded).
    For each numeric feature f:
        f_adj = f - OLS( onehot(S) -> f )
    Returns a transformed copy of df, preserving dtypes where feasible.
    """
    from numpy.linalg import lstsq

    if sensitive_col not in df.columns:
        return df.copy()

    out = df.copy()
    # design matrix for S (one-hot, drop_first to avoid multicollinearity)
    S = pd.get_dummies(df[sensitive_col].astype(str), drop_first=True)
    if S.shape[1] == 0:
        return out

    # Add intercept
    S_design = pd.concat([pd.Series(1.0, index=S.index, name="__intercept__"), S], axis=1).astype(float)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        y = df[col].astype(float)
        # Solve least squares: S_design * beta ≈ y
        try:
            beta, *_ = lstsq(S_design.values, y.values, rcond=None)
            y_hat = S_design.values @ beta
            out[col] = (y.values - y_hat)
        except Exception:
            # if anything fails, keep original
            out[col] = y
    return out