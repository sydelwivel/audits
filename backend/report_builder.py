# backend/report_builder.py
"""
Report Builder: generates a comprehensive PDF report with
- baseline vs mitigated (reweigh+resample) training
- performance & fairness metrics (Accuracy, Precision, Recall, F1, ROC_AUC, DPD, EOD)
- before/after comparison tables
- confusion matrices, ROC curves
- drift one-liner + illustrative plot
- composite scorecard (0-100) combining performance & fairness

Safe to import from Streamlit. Uses only matplotlib (no seaborn).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)


@dataclass
class ReportInputs:
    df: pd.DataFrame
    target: str
    sensitive: str
    test_size: float = 0.3
    random_state: int = 42
    exclude_sensitive_from_features: bool = True


@dataclass
class ReportOutputs:
    pdf_path: str
    performance_compare: pd.DataFrame
    baseline_metrics: Dict[str, float]
    mitigated_metrics: Dict[str, float]
    baseline_fairness: Dict[str, float]
    mitigated_fairness: Dict[str, float]
    drift_sentence: str
    figures: List[plt.Figure] 


# ----------------------------
# Fairness helpers
# ----------------------------
def demographic_parity_difference(y_pred: pd.Series, s: pd.Series) -> float:
    rates = y_pred.groupby(s).mean()
    return float(rates.max() - rates.min())


def equalized_odds_difference(y_true: pd.Series, y_pred: pd.Series, s: pd.Series) -> float:
    vals = {}
    for grp in s.unique():
        idx = (s == grp)
        yt = y_true[idx]
        yp = y_pred[idx]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        vals[grp] = (tpr, fpr)
    tprs = [v[0] for v in vals.values()]
    fprs = [v[1] for v in vals.values()]
    return float((max(tprs) - min(tprs) + max(fprs) - min(fprs)) / 2.0)


# ----------------------------
# Mitigation helpers
# ----------------------------
def reweighing_weights(y: pd.Series, s: pd.Series) -> pd.Series:
    """Kamiran & Calders-style reweighing (binary S & Y expected)."""
    s_vals = list(s.unique())
    if len(s_vals) != 2:
        # fallback: map most frequent to 0, other to 1
        counts = s.value_counts()
        s_map = {counts.index[0]: 0, counts.index[1]: 1}
    else:
        s_map = {s_vals[0]: 0, s_vals[1]: 1}
    s_bin = s.map(s_map).astype(int)
    y_bin = y.astype(int)

    p_y1 = y_bin.mean()
    p_s1 = s_bin.mean()

    w = np.ones(len(y), dtype=float)
    for sv in [0, 1]:
        for yv in [0, 1]:
            idx = (s_bin == sv) & (y_bin == yv)
            p_sy = idx.mean()
            if p_sy > 0:
                target = (p_s1 if sv == 1 else (1 - p_s1)) * (p_y1 if yv == 1 else (1 - p_y1))
                w[idx] = target / p_sy
    return pd.Series(w, index=y.index)


def resample_with_weights(X: pd.DataFrame, y: pd.Series, w: pd.Series, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    probs = (w / w.sum()).values
    idx = rng.choice(np.arange(n), size=n, replace=True, p=probs)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


# ----------------------------
# Model + metrics
# ----------------------------
def train_rf(
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    Xte: pd.DataFrame,
    n_estimators: int = 250,
    sample_weight: Optional[pd.Series] = None,
    random_state: int = 42,
):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    if sample_weight is not None:
        clf.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    return clf, preds, proba


def metrics_summary(y_true: pd.Series, y_pred: pd.Series, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return out


# ----------------------------
# PDF assembly
# ----------------------------
def score_from_metrics(acc: float, dpd: float, eod: float) -> int:
    # 0..100 score: 60% performance, 40% fairness (penalize high dpd/eod)
    fairness_penalty = (dpd + eod) / 2.0
    fairness_term = max(0.0, 1.0 - fairness_penalty)
    return int(round(100 * (0.6 * acc + 0.4 * fairness_term)))


def _plot_confusion(y_true: pd.Series, y_pred: np.ndarray, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha="center", va="center")
    return fig


def build_report(inputs: ReportInputs, pdf_path: str = "reports/ethical_ai_report.pdf") -> ReportOutputs:
    df = inputs.df.copy()
    target = inputs.target
    sensitive = inputs.sensitive

    # Prepare features/labels
    X_all = df.drop(columns=[target])
    if inputs.exclude_sensitive_from_features and sensitive in X_all.columns:
        X_all = X_all.drop(columns=[sensitive])
    y_all = df[target].copy()

    # Binarize labels if needed (strings → choose last label as positive)
    if not set(pd.Series(y_all).dropna().unique()) <= {0, 1}:
        uniq = pd.Series(y_all.dropna().unique())
        pos = uniq.iloc[-1]
        y_all = y_all.apply(lambda v: 1 if v == pos else 0).astype(int)

    # One-hot encode features
    X_all = pd.get_dummies(X_all, drop_first=True)

    # Sensitive series (aligned)
    s_all = df[sensitive].astype(str)

    # Split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_all, y_all, s_all, test_size=inputs.test_size, random_state=inputs.random_state, stratify=y_all
    )

    # Baseline
    base_clf, base_pred, base_proba = train_rf(X_train, y_train, X_test, random_state=inputs.random_state)
    base_metrics = metrics_summary(y_test, base_pred, base_proba)
    base_dpd = demographic_parity_difference(pd.Series(base_pred, index=y_test.index), s_test.reset_index(drop=True))
    base_eod = equalized_odds_difference(y_test.reset_index(drop=True), pd.Series(base_pred).reset_index(drop=True), s_test.reset_index(drop=True))

    # Mitigation: reweigh + resample
    w_train = reweighing_weights(y_train.reset_index(drop=True), s_train.reset_index(drop=True))
    X_rw, y_rw = resample_with_weights(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        w_train,
        random_state=inputs.random_state,
    )
    mit_clf, mit_pred, mit_proba = train_rf(X_rw, y_rw, X_test, random_state=inputs.random_state)
    mit_metrics = metrics_summary(y_test, mit_pred, mit_proba)
    mit_dpd = demographic_parity_difference(pd.Series(mit_pred, index=y_test.index), s_test.reset_index(drop=True))
    mit_eod = equalized_odds_difference(y_test.reset_index(drop=True), pd.Series(mit_pred).reset_index(drop=True), s_test.reset_index(drop=True))

    # Comparison table
    perf_compare = pd.DataFrame([
        {"Model": "Baseline", **{k: round(v, 4) for k, v in base_metrics.items()}, "DPD": round(base_dpd, 4), "EOD": round(base_eod, 4)},
        {"Model": "Mitigated", **{k: round(v, 4) for k, v in mit_metrics.items()}, "DPD": round(mit_dpd, 4), "EOD": round(mit_eod, 4)},
    ])

    # Drift (one-liner + illustrative plot)
    drift_sentence = "Data drift occurs when the statistical properties of input data change over time, causing model performance to degrade."

    # Build figures (one chart per figure; default colors)
    figs: List[plt.Figure] = []

    # Performance bars
    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        if metric in base_metrics and metric in mit_metrics:
            fig = plt.figure()
            plt.bar(["Baseline", "Mitigated"], [base_metrics[metric], mit_metrics[metric]])
            plt.title(f"{metric} Before vs After")
            plt.ylabel(metric)
            plt.ylim(0, 1)
            figs.append(fig)

    # Fairness bars
    fig = plt.figure()
    plt.bar(["DPD-Before", "DPD-After"], [base_dpd, mit_dpd])
    plt.title("Demographic Parity Difference (lower is better)")
    plt.ylabel("Difference")
    plt.ylim(0, 1)
    figs.append(fig)

    fig = plt.figure()
    plt.bar(["EOD-Before", "EOD-After"], [base_eod, mit_eod])
    plt.title("Equalized Odds Difference (lower is better)")
    plt.ylabel("Difference")
    plt.ylim(0, 1)
    figs.append(fig)

    # Confusion matrices
    figs.append(_plot_confusion(y_test, base_pred, "Confusion Matrix (Baseline)"))
    figs.append(_plot_confusion(y_test, mit_pred, "Confusion Matrix (Mitigated)"))

    # ROC curves
    try:
        fpr_b, tpr_b, _ = roc_curve(y_test, base_proba)
        fig = plt.figure()
        plt.plot(fpr_b, tpr_b, label="Baseline ROC")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve (Baseline)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        figs.append(fig)
    except Exception:
        pass

    try:
        fpr_m, tpr_m, _ = roc_curve(y_test, mit_proba)
        fig = plt.figure()
        plt.plot(fpr_m, tpr_m, label="Mitigated ROC")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve (Mitigated)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        figs.append(fig)
    except Exception:
        pass

    # Composite scorecard
    baseline_score = score_from_metrics(base_metrics.get("Accuracy", 0.0), base_dpd, base_eod)
    mitigated_score = score_from_metrics(mit_metrics.get("Accuracy", 0.0), mit_dpd, mit_eod)
    scorecard_df = pd.DataFrame({
        "Model": ["Baseline", "Mitigated"],
        "CompositeScore(0-100)": [baseline_score, mitigated_score],
        "Accuracy": [round(base_metrics.get("Accuracy", 0.0), 4), round(mit_metrics.get("Accuracy", 0.0), 4)],
        "DPD": [round(base_dpd, 4), round(mit_dpd, 4)],
        "EOD": [round(base_eod, 4), round(mit_eod, 4)],
    })

    # Write PDF
    import os
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.1, 0.9, "Ethical AI Project Report", fontsize=18, weight="bold")
        plt.text(0.1, 0.84, "Comprehensive Analysis & Findings", fontsize=12)
        plt.text(0.1, 0.80, f"Target: {target} | Sensitive: {sensitive}", fontsize=10)
        plt.text(0.1, 0.75, "Summary:", fontsize=12, weight="bold")
        plt.text(0.1, 0.71, f"- Baseline Accuracy: {base_metrics.get('Accuracy', 0.0):.3f} | Mitigated Accuracy: {mit_metrics.get('Accuracy', 0.0):.3f}")
        plt.text(0.1, 0.68, f"- DPD Before: {base_dpd:.3f} → After: {mit_dpd:.3f}")
        plt.text(0.1, 0.65, f"- EOD Before: {base_eod:.3f} → After: {mit_eod:.3f}")
        pdf.savefig(fig); plt.close(fig)

        # Comparison table
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.1, 0.95, "Before/After Performance & Fairness Comparison", fontsize=14, weight="bold")
        plt.text(0.05, 0.88, perf_compare.to_string(index=False), family="monospace", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # Scorecard
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.1, 0.95, "Scorecard", fontsize=14, weight="bold")
        plt.text(0.05, 0.88, scorecard_df.to_string(index=False), family="monospace", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # Drift page
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.1, 0.95, "Data Drift", fontsize=14, weight="bold")
        plt.text(0.1, 0.90, drift_sentence, fontsize=11)
        plt.text(0.1, 0.86, "Illustration: distribution shift between training and new data.", fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Add all figures
        for f in figs:
            pdf.savefig(f)
            plt.close(f)

    return ReportOutputs(
        pdf_path=pdf_path,
        performance_compare=perf_compare,
        baseline_metrics={**base_metrics, "DPD": base_dpd, "EOD": base_eod},
        mitigated_metrics={**mit_metrics, "DPD": mit_dpd, "EOD": mit_eod},
        baseline_fairness={"DPD": base_dpd, "EOD": base_eod},
        mitigated_fairness={"DPD": mit_dpd, "EOD": mit_eod},
        drift_sentence=drift_sentence,
        figures=figs   # return all figures for Streamlit display
    )
