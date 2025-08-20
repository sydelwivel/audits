# frontend/app.py

"""

EthixAI - Streamlit frontend (final, patched + Debiasing, Light Theme)

- Light professional UI (blue accents), accessible contrast across all sections

- Readable tables: white background, subtle striping, dark text (Proxy Bias & Privacy Risk use HTML table styling)

- Preserves all original features and pages

- Adds REAL bias mitigation (Reweighing) with dataset resampling and sample-weight training

- Synthetic generation verification + "already balanced" guardrail (unchanged)

- Community backend for crowdsourcing (UPDATED with voting)

- Train & Evaluate: regression handled in-app; classification supports sample weights (unchanged)

- Fairness Audit: one-line explanations for DPD & EOD (unchanged)

"""



import sys

import os

from pathlib import Path

import io

import pickle

import tempfile

from datetime import datetime



import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor





if "page" not in st.session_state:

    st.session_state["page"] = "Dashboard"





# ---------------------------

# Custom CSS for UI

# ---------------------------

st.markdown("""

<style>

    /* Main container and body */

    .stApp {

        background-color: #F8F9FA;

        color: #333333;

        font-family: 'Segoe UI', sans-serif;

    }

    /* Sidebar */

    .css-1d391kg {

        background-color: #FFFFFF;

        padding: 1rem;

        box-shadow: 0 4px 6px rgba(0,0,0,0.05);

    }

    .css-1d391kg .css-1aumxav {

        color: #2D58A0;

    }

    /* Headings and Titles */

    h1 {

        color: #2D58A0;

        font-size: 2.5em;

        font-weight: 600;

        margin-bottom: 0.5em;

    }

    h2 {

        color: #333333;

        font-size: 1.8em;

        border-bottom: 2px solid #DDDDDD;

        padding-bottom: 0.2em;

    }

    h3 {

        color: #2D58A0;

        font-size: 1.4em;

    }

    .section-title {

        font-size: 1.8em;

        font-weight: 500;

        color: #2D58A0;

        border-bottom: 2px solid #DDDDDD;

        padding-bottom: 0.2em;

        margin-bottom: 1rem;

    }

    .subtle {

        color: #6c757d;

        font-size: 0.9em;

        margin-bottom: 2rem;

    }

    .metric-spacer {

        height: 10px;

    }

    /* Cards */

    .card {

        background-color: #FFFFFF;

        border-radius: 12px;

        padding: 2rem;

        box-shadow: 0 4px 12px rgba(0,0,0,0.08);

        margin-bottom: 2rem;

    }

    /* Buttons */

    .stButton>button {

        background-color: #2D58A0;

        color: white;

        border-radius: 8px;

        border: none;

        padding: 12px 24px;

        font-size: 1em;

        transition: all 0.2s ease-in-out;

    }

    .stButton>button:hover {

        background-color: #1a4175;

        transform: translateY(-2px);

        box-shadow: 0 2px 6px rgba(0,0,0,0.1);

    }

    /* Tables (Proxy Bias & Privacy Risk) */

    .styled-table {

        border-collapse: collapse;

        margin: 25px 0;

        font-size: 0.9em;

        min-width: 100%;

        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);

    }

    .styled-table thead tr {

        background-color: #2D58A0;

        color: #ffffff;

        text-align: left;

    }

    .styled-table th, .styled-table td {

        padding: 12px 15px;

    }

    .styled-table tbody tr {

        border-bottom: 1px solid #dddddd;

    }

    .styled-table tbody tr:nth-of-type(even) {

        background-color: #f3f3f3;

    }

    .styled-table tbody tr:last-of-type {

        border-bottom: 2px solid #2D58A0;

    }

    .risk-badge {

        padding: 4px 8px;

        border-radius: 12px;

        font-size: 0.8em;

        font-weight: bold;

        color: white;

    }

    .risk-low {

        background-color: #28a745;

    }

    .risk-mid {

        background-color: #ffc107;

    }

    .risk-high {

        background-color: #dc3545;

    }

</style>

""", unsafe_allow_html=True)



# ---------------------------

# Ensure backend folder is on sys.path

# ---------------------------

HERE = Path(__file__).resolve().parent

PROJECT_ROOT = HERE.parent  # assumes frontend/ and backend/ are siblings

BACKEND_PATH = PROJECT_ROOT / "backend"

if str(BACKEND_PATH) not in sys.path:

    sys.path.insert(0, str(BACKEND_PATH))



# ---------------------------

# Import backend modules (graceful)

# ---------------------------

_missing = []

def _try_import(name):

    try:

        module = __import__(name)

        return module

    except Exception:

        _missing.append(name)

        return None



audit = _try_import("audit")

privacy = _try_import("privacy")

synthetic = _try_import("synthetic")

retrain = _try_import("retrain")

# scorecard = _try_import("scorecard") # Removed scorecard import

drift = _try_import("drift")

explain = _try_import("explain")

simulator = _try_import("simulator")

community = _try_import("community")

mitigation = _try_import("mitigation")   # Debiasing backend (optional, if present)



# ---------------------------

# Helpers

# ---------------------------

# -------- Session pipeline helpers (NEW) --------

def set_raw_dataset(df: pd.DataFrame):

    # Keep an immutable copy of the original upload

    if "raw_dataset" not in st.session_state:

        st.session_state["raw_dataset"] = df.copy()



def set_debiased_dataset(df: pd.DataFrame, meta: dict):

    st.session_state["debiased_dataset"] = df.copy()

    st.session_state["debiased_meta"] = meta



def detect_sensitive_attributes(df: pd.DataFrame):

    sensitive_keywords = [

        "gender", "sex", "race", "ethnicity", "age",

        "religion", "disability", "nationality",

        "marital_status", "income", "sexual_orientation"

    ]

    detected = []

    for col in df.columns:

        low = col.lower().replace(" ", "_")

        for kw in sensitive_keywords:

            if kw in low:

                detected.append(col)

                break

    return detected



def binarize_labels(series: pd.Series):

    s = pd.Series(series).copy()

    uniq = pd.Series(s.dropna().unique())

    set_uniq = set(uniq)

    if set_uniq <= {0,1} or set_uniq <= {-1,1}:

        return s.astype(int).replace({-1:1})

    # choose positive label heuristically

    if any(isinstance(x, str) for x in uniq):

        for val in uniq:

            lv = str(val).lower()

            if ">" in lv or "yes" in lv or lv in ("1","true","t","approved","success","positive"):

                pos = val

                break

        else:

            pos = uniq.iloc[-1]

    else:

        pos = max(uniq)

    return s.apply(lambda x: 1 if x == pos else 0).astype(int)



def df_to_csv_bytes(df: pd.DataFrame) -> bytes:

    return df.to_csv(index=False).encode("utf-8")



def start_card():

    st.markdown("<div class='card'>", unsafe_allow_html=True)



def end_card():

    st.markdown("</div>", unsafe_allow_html=True)



def class_balance_info(s: pd.Series):

    counts = s.value_counts(dropna=False)

    if len(counts) < 2:

        return {

            "is_valid": False,

            "message": "Target has only one unique value — balancing not applicable.",

            "counts": counts,

            "imbalance_ratio": 0.0

        }

    max_c = counts.max()

    min_c = counts.min()

    imb = (max_c - min_c) / max(1.0, float(max_c))

    return {

        "is_valid": True,

        "message": "",

        "counts": counts,

        "imbalance_ratio": float(imb),

    }



# Function to calculate overall model score (for 'Train & Evaluate' section)

def _calculate_score(metrics: dict):

    if not metrics:

        return 0.0

    if "Accuracy" in metrics and "F1" in metrics:

        return (metrics["Accuracy"] * 0.5 + metrics["F1"] * 0.5) * 100

    if "R2" in metrics:

        return max(0.0, min(1.0, metrics["R2"])) * 100

    return np.mean(list(metrics.values())) * 100



# ---------------------------

# Sidebar / Navigation

# ---------------------------

logo_path = os.environ.get("ETHIXAI_LOGO_PATH", "")  # optional environment variable

if logo_path and os.path.exists(logo_path):

    st.sidebar.image(logo_path, width=140)



st.sidebar.title("EthixAI")

st.sidebar.caption("Ethical AI Auditor")



# Initialize page state in session_state

if "page" not in st.session_state:

    st.session_state.page = "Dashboard"



menu = st.sidebar.radio("Go to", [

    "Dashboard",

    "Fairness Audit",

    "Debiasing",

    "Proxy Bias",

    "Privacy Risk",

    "Synthetic Data",

    "Train & Evaluate",

    "Simulator",

    "Drift",

    "Community"

], index=["Dashboard","Fairness Audit","Debiasing","Proxy Bias","Privacy Risk","Synthetic Data",

          "Train & Evaluate","Simulator","Drift","Community"].index(st.session_state.get("page", "Dashboard")))



st.session_state.page = menu



if _missing:

    st.sidebar.error(f"Missing backend modules: {_missing}. Some features may be disabled.")



# ---------------------------

# Dataset upload / load

# ---------------------------

st.markdown("<div class='h1'>EthixAI — Complete Ethical AI Auditor</div>", unsafe_allow_html=True)

st.markdown("<div class='subtle'>Upload a dataset to start auditing fairness, privacy, proxies, and more.</div>", unsafe_allow_html=True)



uploaded = st.file_uploader("Upload dataset (CSV / Excel / JSON)", type=["csv","xlsx","json"])

if uploaded:

    try:

        if uploaded.name.endswith(".csv"):

            df = pd.read_csv(uploaded)

        elif uploaded.name.endswith(".xlsx"):

            df = pd.read_excel(uploaded)

        else:

            df = pd.read_json(uploaded)

        st.session_state["df"] = df

        # clear any prior weights when a new dataset is loaded

        st.session_state.pop("reweigh_weights", None)

        st.session_state.pop("weights_info", None)

        st.success("Dataset loaded.")

                # NEW: store immutable original for true "before" in pipeline

        set_raw_dataset(df)

        # Reset downstream artifacts when a new dataset is uploaded

        for k in ["debiased_dataset","debiased_meta","raw_metrics","debiased_metrics",

                  "raw_model","debiased_model","performance_compare"]:

            st.session_state.pop(k, None)



    except Exception as e:

        st.error(f"Failed to load dataset: {e}")



if "df" not in st.session_state:

    st.session_state["df"] = None



df = st.session_state["df"]



# ---------------------------

# Dashboard page

# ---------------------------

if st.session_state.page == "Dashboard":

    start_card()

    st.markdown("<div class='section-title'>Dataset Preview & Quick Actions</div>", unsafe_allow_html=True)

    if df is None:

        st.info("Upload a dataset to begin auditing.")

    else:

        st.dataframe(df.head(), use_container_width=True)

        auto_sensitive = detect_sensitive_attributes(df)

        if auto_sensitive:

            st.info(f"Auto-detected sensitive columns: {auto_sensitive}")

        col1, col2, col3 = st.columns([1,1,1])

        with col1:

            if st.button("Run Fairness Audit"):

                st.session_state.page = "Fairness Audit"

                st.rerun()

        with col2:

            if st.button("Debias Dataset"):

                st.session_state.page = "Debiasing"

                st.rerun()

        with col3:

            if st.button("Train Model (quick)"):

                st.session_state.page = "Train & Evaluate"

                st.rerun()

    end_card()



# ---------------------------

# Fairness Audit page

# ---------------------------

if st.session_state.page == "Fairness Audit":

    start_card()

    st.markdown("<div class='section-title'>Fairness Audit</div>", unsafe_allow_html=True)

    st.caption("• Demographic Parity Difference: difference in **positive prediction rates** between groups (0 = equal rates).  • Equalized Odds Difference: difference in **error rates (TPR/FPR)** between groups (0 = equal errors).")

    if df is None:

        st.warning("Upload dataset first.")

    else:

                # Ensure original is stored for pipeline flow

        set_raw_dataset(df)



        cols = list(df.columns)

        default_target_idx = min(len(cols)-1, 0) if cols else 0

        target = st.selectbox("Target column", cols, index=default_target_idx, key="audit_target")

        auto_sensitive = detect_sensitive_attributes(df)

        sensitive = st.selectbox("Sensitive attribute", auto_sensitive + cols if auto_sensitive else cols, key="audit_sensitive")



        col_a, col_b = st.columns([1,1])

        with col_a:

            run_btn = st.button("Run Fairness Audit")

        with col_b:

            debias_shortcut = st.button("Mitigate Bias (go to Debiasing)")



        if debias_shortcut:

            st.session_state.page = "Debiasing"

            st.rerun()



        if run_btn:

            if audit is None:

                st.error("Audit backend missing.")

            else:

                # determine y_true and y_pred

                y_true = binarize_labels(df[target])



                pred_cols = [c for c in df.columns if "pred" in c.lower() or c.lower().endswith("_pred")]

                prob_cols = [c for c in df.columns if ("prob" in c.lower()) or ("score" in c.lower())]



                if pred_cols:

                    y_pred = binarize_labels(df[pred_cols[0]])

                    st.info(f"Using predictions from column: {pred_cols[0]}")

                elif prob_cols:

                    y_pred = (pd.to_numeric(df[prob_cols[0]], errors="coerce") >= 0.5).astype(int)

                    st.info(f"Using probabilities from column: {prob_cols[0]} (threshold 0.5)")

                else:

                    # fallback: train internal model excluding sensitive column

                    try:

                        feat_cols = [c for c in df.columns if c not in [target, sensitive]]

                        if len(feat_cols) == 0:

                            st.error("No features available for training fallback model.")

                            y_pred = y_true

                        else:

                            X = pd.get_dummies(df[feat_cols], drop_first=True)

                            y = y_true

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                            clf = RandomForestClassifier(n_estimators=100, random_state=42)

                            clf.fit(X_train, y_train)

                            y_pred = pd.Series(clf.predict(X), index=df.index)

                            st.info("Trained internal model (sensitive excluded) for audit.")

                    except Exception as e:

                        st.error(f"Fallback model failed: {e}")

                        y_pred = y_true



                metrics = audit.run_fairness_audit(y_true, y_pred, df[sensitive])



                def interpret(results):

                    out = {}

                    for metric, val in results.items():

                        a = abs(val)

                        if a <= 0.05:

                            level = "✅ Fair"; col = "#166534"; expl = "Little to no measurable bias."

                        elif a <= 0.15:

                            level = "⚠️ Mild Bias"; col = "#92400e"; expl = "Small differences — review."

                        else:

                            level = "❌ Significant Bias"; col = "#991b1b"; expl = "Large differences — action required."

                        out[metric] = {"value": round(val,4), "level": level, "color": col, "explanation": expl}

                    return out



                interp = interpret(metrics)

                for m, info in interp.items():

                    st.markdown(f"**{m}:** {info['value']}   {info['level']}")

                    st.markdown(f"<small style='color:{info['color']}'>{info['explanation']}</small>", unsafe_allow_html=True)

                    st.progress(min(abs(info['value']), 1.0))

                    st.markdown("<div class='metric-spacer'></div>", unsafe_allow_html=True)



                # group-wise stats

                st.subheader("Group statistics")

                grp = df[sensitive].astype(str)

                pos_rate = pd.Series(y_pred).groupby(grp).mean().rename("Positive Rate")

                counts = grp.value_counts().rename("Count")

                gdf = pd.concat([counts, pos_rate], axis=1).fillna(0).reset_index().rename(columns={"index":"Group"})

                st.table(gdf)

    end_card()



# ---------------------------

# Debiasing page

# ---------------------------

if st.session_state.page == "Debiasing":

    start_card()

    st.markdown("<div class='section-title'>Debias Dataset (Pre-processing)</div>", unsafe_allow_html=True)

    st.caption(

        "Reweighing makes **S (sensitive)** and **Y (label)** more independent. "

        "Create a *debiased dataset* (via weighted resampling) or compute *sample weights* to use during model training."

    )

    if df is None:

        st.warning("Upload dataset first.")

    else:

        cols = list(df.columns)

        sensitive_candidates = detect_sensitive_attributes(df)

        sensitive = st.selectbox(

            "Sensitive attribute",

            sensitive_candidates + cols if sensitive_candidates else cols,

            key="deb_sensitive"

        )

        target = st.selectbox("Target column", cols, key="deb_target")



        strategy = st.selectbox(

            "Mitigation strategy",

            ["Reweigh & Resample (recommended)", "Reweigh Only (use in training)", "Decorrelate Numeric Features"],

            index=0

        )

        strat_key = {

            "Reweigh & Resample (recommended)": "reweigh_resample",

            "Reweigh Only (use in training)": "reweigh_only",

            "Decorrelate Numeric Features": "decorrelate_numeric"

        }[strategy]



        replace_now = st.checkbox("Replace current dataset with debiased output", value=False)

        run = st.button("Run Debiasing")



        if run:

            if mitigation is None:

                st.error("Mitigation backend missing.")

            else:

                try:

                    deb_df, weights, report = mitigation.debias_dataset(

                        df, sensitive, target, strategy=strat_key, random_state=42

                    )



                    st.success("Debiasing completed.")

                    st.write("**Report:**", report)



                    # Save debiased dataset and metadata in session_state for downstream use

                    set_debiased_dataset(

                        deb_df,

                        {

                            "strategy": strat_key,

                            "sensitive": sensitive,

                            "target": target,

                            "timestamp": datetime.utcnow().isoformat()

                        }

                    )



                    # Reset downstream models/metrics when new debiased data is created

                    for k in ["debiased_metrics", "debiased_model", "performance_compare"]:

                        st.session_state.pop(k, None)



                    # Preview debiased data

                    st.write("#### Preview (debiased)")

                    st.dataframe(deb_df.head(10), use_container_width=True)



                    # Download button for debiased dataset CSV export

                    st.download_button(

                        label="Download debiased CSV",

                        data=df_to_csv_bytes(deb_df),

                        file_name="debiased_dataset.csv",

                        mime="text/csv"

                    )



                    # Store weights for training page

                    st.session_state["reweigh_weights"] = weights

                    st.session_state["weights_info"] = {

                        "strategy": strat_key,

                        "sensitive": sensitive,

                        "target": target,

                        "mean_weight": float(np.mean(weights.values)) if len(weights) else 1.0

                    }



                    # Optionally replace current session dataset with debiased output

                    if replace_now:

                        st.session_state["df"] = deb_df

                        st.info("Session dataset replaced with debiased output.")



                except Exception as e:

                    st.error(f"Debiasing failed: {e}")

    end_card()





# ---------------------------

# Proxy Bias page

# ---------------------------

if st.session_state.page == "Proxy Bias":

    start_card()

    st.markdown("<div class='section-title'>Proxy Bias Detection</div>", unsafe_allow_html=True)

    st.caption("A **proxy** is a feature that stands in for a sensitive attribute. We flag features with a strong association to the chosen sensitive column (0% = no link, 100% = very strong link).")

    if df is None:

        st.warning("Upload dataset first.")

    else:

        sensitive_candidates = detect_sensitive_attributes(df)

        sensitive = st.selectbox("Sensitive attribute", sensitive_candidates + list(df.columns) if sensitive_candidates else list(df.columns))

        top_n = st.number_input("Top N proxies", min_value=1, max_value=20, value=5)

        if st.button("Detect Proxy Bias"):

            if audit is None:

                st.error("Audit backend missing.")

            else:

                # Support both backends:

                # 1) advanced list of dicts

                # 2) simple numeric correlation Series/df/dict

                try:

                    proxies = audit.detect_proxy_bias(df, sensitive)

                except TypeError:

                    proxies = audit.detect_proxy_bias(df, sensitive, int(top_n))



                rows = []



                if isinstance(proxies, list):

                    for i, p in enumerate(proxies[:int(top_n)]):

                        strength_pct = float(p.get("strength", 0)) * 100.0

                        risk = p.get("risk_level", "Low")

                        badge_cls = "risk-low"

                        if "Strong" in risk or "High" in risk:

                            badge_cls = "risk-high"

                        elif "Mild" in risk or "Medium" in risk:

                            badge_cls = "risk-mid"

                        rows.append({

                            "Rank": i+1,

                            "Feature": p.get("column", ""),

                            "Association (0–100%)": round(strength_pct, 1),

                            "Risk": f"<span class='risk-badge {badge_cls}'>{risk}</span>",

                            "Why this matters": p.get("reason", "Associated with the sensitive attribute."),

                            "Suggested Action": p.get("suggestion", "Consider removing or transforming this feature.")

                        })

                elif isinstance(proxies, (pd.Series, pd.DataFrame, dict)):

                    if isinstance(proxies, dict):

                        ser = pd.Series(proxies)

                    else:

                        ser = proxies if isinstance(proxies, pd.Series) else proxies.squeeze()

                    ser = ser.drop(labels=[sensitive], errors="ignore")

                    ser = ser.abs().sort_values(ascending=False)[:int(top_n)]

                    for i, (feat, val) in enumerate(ser.items()):

                        strength_pct = float(val) * 100.0

                        if strength_pct >= 30:

                            risk, badge_cls = "Strong Proxy Risk", "risk-high"

                            why = "Feature shows a strong association with the sensitive attribute; may act as a stand-in."

                            sugg = "Audit usage; consider removing, bucketing, or adversarial de-correlation."

                        elif strength_pct >= 10:

                            risk, badge_cls = "Mild Proxy Risk", "risk-mid"

                            why = "Moderate association with the sensitive attribute."

                            sugg = "Monitor impact; consider regularization or partial removal."

                        else:

                            risk, badge_cls = "Low Risk", "risk-low"

                            why = "Weak association with the sensitive attribute."

                            sugg = "Usually safe; keep monitoring."

                        rows.append({

                            "Rank": i+1,

                            "Feature": feat,

                            "Association (0–100%)": round(strength_pct, 1),

                            "Risk": f"<span class='risk-badge {badge_cls}'>{risk}</span>",

                            "Why this matters": why,

                            "Suggested Action": sugg

                        })

                else:

                    st.info("No proxy associations found.")

                    rows = []



                if rows:

                    df_view = pd.DataFrame(rows)

                    def df_to_html_table(dfx):

                        return dfx.to_html(index=False, classes="styled-table", escape=False)

                    st.markdown(df_to_html_table(df_view), unsafe_allow_html=True)

    end_card()



# ---------------------------

# Privacy Risk page

# ---------------------------

if st.session_state.page == "Privacy Risk":

    start_card()

    st.markdown("<div class='section-title'>Privacy Risk Audit (Top 5)</div>", unsafe_allow_html=True)

    if df is None:

        st.warning("Upload dataset first.")

    else:

        if st.button("Run Privacy Audit"):

            if privacy is None:

                st.error("Privacy backend missing.")

            else:

                risky = privacy.reidentifiable_features(df, top_n=5)

                if not risky:

                    st.success("No high-risk combos found (thresholds applied).")

                else:

                    rows = []

                    for i, r in enumerate(risky):

                        rows.append({

                            "Rank": i+1,

                            "Combination": ", ".join(r["combination"]),

                            "Uniqueness (%)": f"{r['unique_ratio']*100:.1f}",

                            "Why this matters": r["reason"],

                            "Suggested Action": r["suggestion"]

                        })

                    pdf = pd.DataFrame(rows)

                    st.markdown(pdf.to_html(index=False, classes="styled-table"), unsafe_allow_html=True)

    end_card()



# ---------------------------

# Synthetic Data page

# ---------------------------

if st.session_state.page == "Synthetic Data":

    start_card()

    st.markdown("<div class='section-title'>Synthetic Data Generation & Verification</div>", unsafe_allow_html=True)

    st.caption("Use this to balance a **categorical target** when classes are notably uneven. If your dataset is already close to balanced, we’ll recommend leaving it as-is.")

    if df is None:

        st.warning("Upload dataset first.")

    else:

        cols = list(df.columns)

        target_col = st.selectbox("Select target column to balance", cols)

        sensitive_candidates = detect_sensitive_attributes(df)

        sensitive_to_check = st.selectbox("Sensitive column to verify retention (optional)", ["None"] + sensitive_candidates) if sensitive_candidates else "None"

        top_n_preview = st.number_input("Preview rows after generation", min_value=3, max_value=50, value=5)



        y = df[target_col]

        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:

            st.warning("The selected target looks **numeric/continuous**. Balancing is intended for classification targets. Please pick a categorical/binary target (e.g., 'income').")

            proceed_balancing = False

        else:

            bal = class_balance_info(y.astype(str))

            proceed_balancing = True

            if bal["is_valid"]:

                st.write("#### Class distribution (original)")

                st.write(bal["counts"])

                tolerance = 0.15

                if bal["imbalance_ratio"] <= tolerance:

                    st.info("Your dataset already looks **balanced enough** (within ~15% between classes). In many cases, this is fine — you don't need to synthesize more data.")

                    override = st.checkbox("Proceed with synthetic balancing anyway")

                    if not override:

                        proceed_balancing = False



        if st.button("Generate synthetic dataset"):

            if synthetic is None:

                st.error("Synthetic backend missing.")

            else:

                if not proceed_balancing:

                    st.warning("Skipped synthetic generation based on the assessment above.")

                else:

                    try:

                        syn_df = synthetic.generate_synthetic_data(df, target_col)

                        if syn_df is None:

                            st.error("Synthetic generator returned nothing.")

                        else:

                            st.success("Synthetic dataset generated.")

                            st.write("#### Preview (synthetic)")

                            st.dataframe(syn_df.head(top_n_preview), use_container_width=True)

                            after_counts = syn_df[target_col].value_counts(dropna=False)

                            st.write("#### Class distribution (synthetic)")

                            st.write(after_counts)

                            if sensitive_to_check and sensitive_to_check != "None":

                                retained = sensitive_to_check in syn_df.columns

                                st.write(f"Sensitive column '{sensitive_to_check}' retained in synthetic dataset: {retained}")

                                if retained:

                                    st.write(syn_df[sensitive_to_check].value_counts(dropna=False))

                                else:

                                    st.error(f"Sensitive column '{sensitive_to_check}' missing in synthetic output!")

                            csv_bytes = df_to_csv_bytes(syn_df)

                            st.download_button("Download synthetic CSV", data=csv_bytes, file_name="synthetic_dataset.csv", mime="text/csv")

                            st.session_state["df"] = syn_df

                    except Exception as e:

                        st.error(f"Synthetic generation failed: {e}")

    end_card()





# ---------------------------

# Train & Evaluate page

# ---------------------------

if st.session_state.page == "Train & Evaluate":

    start_card()

    st.markdown("<div class='section-title'>Train & Evaluate Model</div>", unsafe_allow_html=True)



    # --------------- DISK LOAD: Recover datasets on page load if missing ---------------

    # Load raw dataset if missing

    if "raw_dataset" not in st.session_state and os.path.isfile("cache_raw_dataset.csv"):

        st.session_state["raw_dataset"] = pd.read_csv("cache_raw_dataset.csv")

    # Load debiased dataset if missing

    if "debiased_dataset" not in st.session_state and os.path.isfile("cache_debiased_dataset.csv"):

        st.session_state["debiased_dataset"] = pd.read_csv("cache_debiased_dataset.csv")

    # Load metrics if missing

    if "raw_metrics" not in st.session_state and os.path.isfile("cache_raw_metrics.pkl"):

        with open("cache_raw_metrics.pkl", "rb") as f:

            st.session_state["raw_metrics"] = pickle.load(f)

    if "debiased_metrics" not in st.session_state and os.path.isfile("cache_debiased_metrics.pkl"):

        with open("cache_debiased_metrics.pkl", "rb") as f:

            st.session_state["debiased_metrics"] = pickle.load(f)

    if "performance_compare" not in st.session_state and os.path.isfile("cache_perf_compare.pkl"):

        with open("cache_perf_compare.pkl", "rb") as f:

            st.session_state["performance_compare"] = pickle.load(f)



    main_df = st.session_state.get("df", None)

    raw_df = st.session_state.get("raw_dataset", None)

    debiased_df = st.session_state.get("debiased_dataset", None)



    # ----------------------------------- Upload debiased CSV -----------------------------------

    st.markdown("#### Optional: Upload debiased dataset CSV (from Debiasing page)")

    uploaded_debiased_file = st.file_uploader(

        "Upload debiased CSV",

        type=["csv"],

        key="debiased_csv_upload"

    )

    if uploaded_debiased_file is not None:

        uploaded_df = pd.read_csv(uploaded_debiased_file)

        st.session_state["debiased_dataset"] = uploaded_df

        debiased_df = uploaded_df

        uploaded_df.to_csv("cache_debiased_dataset.csv",index=False)

        st.success("Debiased dataset uploaded and saved!")



    # --------------------------------- Upload raw CSV (optional, if raw missing) ----------------

    # Only show if session is missing raw

    if (main_df is None and raw_df is None):

        st.markdown("#### Upload Raw Dataset CSV")

        uploaded_raw_file = st.file_uploader("Upload raw CSV", type=["csv"], key="raw_csv_upload")

        if uploaded_raw_file is not None:

            uploaded_raw_df = pd.read_csv(uploaded_raw_file)

            st.session_state["raw_dataset"] = uploaded_raw_df

            raw_df = uploaded_raw_df

            uploaded_raw_df.to_csv("cache_raw_dataset.csv",index=False)

            st.success("Raw dataset uploaded and saved!")

    

    # Wait for dataset upload before proceeding

    if main_df is None and raw_df is None:

        st.warning("Upload dataset first (raw data required).")

        end_card()

    else:

        df = main_df if main_df is not None else raw_df



        cols = list(df.columns)

        target = st.selectbox("Target column", cols)

        problem_type = st.selectbox("Problem type", ["Auto-detect", "Classification", "Regression"])

        exclude_sensitive = st.checkbox("Exclude detected sensitive column from features (recommended)", value=True)

        sensitive_candidates = detect_sensitive_attributes(df)

        sensitive = None

        if sensitive_candidates:

            sensitive = st.selectbox("Detected sensitive attribute (for exclusion/fairness)", ["None"] + sensitive_candidates)

        test_size = st.slider("Test set fraction", min_value=0.1, max_value=0.5, value=0.3, step=0.05)



        use_weights = False

        if st.session_state.get("reweigh_weights") is not None:

            wi = st.session_state.get("weights_info", {})

            st.info(f"Reweighing weights available (strategy: {wi.get('strategy','n/a')}, sensitive: {wi.get('target','?')}). You can train with these weights.")

            use_weights = st.checkbox("Use stored sample weights in training (where supported)", value=True)



        def _train_eval_on_df(full_df, target_col, detected_type, sensitive_to_drop):

            X_all = full_df.drop(columns=[target_col])

            if exclude_sensitive and sensitive_to_drop and sensitive_to_drop != "None" and sensitive_to_drop in X_all.columns:

                X_all = X_all.drop(columns=[sensitive_to_drop])

            y_all = full_df[target_col]

            X_enc = pd.get_dummies(X_all, drop_first=True)

            fit_kwargs = {}

            if use_weights and st.session_state.get("reweigh_weights") is not None:

                w_all = pd.Series(st.session_state["reweigh_weights"]).reindex(full_df.index).fillna(1.0)

                fit_kwargs["sample_weight"] = w_all.loc[X_enc.index]

            if detected_type == "Classification":

                y_enc = binarize_labels(y_all)

                X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=test_size, random_state=42)

                clf = RandomForestClassifier(n_estimators=200, random_state=42)

                try:

                    clf.fit(X_train, y_train, **fit_kwargs)

                except TypeError:

                    clf.fit(X_train, y_train)

                preds = clf.predict(X_test)

                metrics = {

                    "Accuracy": float(accuracy_score(y_test, preds)),

                    "Precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),

                    "Recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),

                    "F1": float(f1_score(y_test, preds, average="weighted", zero_division=0)),

                }

                return metrics, clf, preds, y_test

            else:

                X_train, X_test, y_train, y_test = train_test_split(X_enc, y_all, test_size=test_size, random_state=42)

                reg = RandomForestRegressor(n_estimators=300, random_state=42)

                try:

                    reg.fit(X_train, y_train, **fit_kwargs)

                except TypeError:

                    reg.fit(X_train, y_train)

                preds = reg.predict(X_test)

                mse = mean_squared_error(y_test, preds)

                metrics = {

                    "R2": float(r2_score(y_test, preds)),

                    "MSE": float(mse),

                    "RMSE": float(np.sqrt(mse)),

                    "MAE": float(mean_absolute_error(y_test, preds))

                }

                return metrics, reg, preds, y_test



        if st.button("Train & Evaluate"):

            try:

                if problem_type == "Regression" or (problem_type == "Auto-detect" and pd.api.types.is_numeric_dtype(df[target])):

                    detected = "Regression"

                else:

                    detected = "Classification"

                st.info(f"Detected problem type: {detected}")



                # Train & evaluate on raw dataset (baseline)

                base_df = raw_df if raw_df is not None else df

                raw_metrics, raw_model, raw_preds, raw_ytest = _train_eval_on_df(base_df, target, detected, sensitive if exclude_sensitive else None)

                st.session_state["raw_metrics"] = raw_metrics

                st.session_state["raw_model"] = raw_model



                # Save to disk for absolute persistence

                os.makedirs("cache", exist_ok=True)

                with open("cache/cache_raw_metrics.pkl", "wb") as f:

                    pickle.dump(raw_metrics, f)



                st.subheader("Baseline (Raw dataset)")

                st.json({k: round(v, 4) for k, v in raw_metrics.items()})



                # Train & evaluate on debiased dataset, if available

                if debiased_df is not None:

                    deb_metrics, deb_model, deb_preds, deb_ytest = _train_eval_on_df(debiased_df, target, detected, sensitive if exclude_sensitive else None)

                    st.session_state["debiased_metrics"] = deb_metrics

                    st.session_state["debiased_model"] = deb_model



                    with open("cache/cache_debiased_metrics.pkl", "wb") as f:

                        pickle.dump(deb_metrics, f)



                    st.subheader("After Debiasing (Debiased dataset)")

                    st.json({k: round(v, 4) for k, v in deb_metrics.items()})



                    perf_compare = pd.DataFrame.from_dict({

                        "Baseline (Raw)": raw_metrics,

                        "After Debiasing": deb_metrics

                    }, orient="index")

                    st.session_state["performance_compare"] = perf_compare

                    with open("cache/cache_perf_compare.pkl", "wb") as f:

                        pickle.dump(perf_compare, f)



                    st.write("### Performance Comparison")

                    st.dataframe(perf_compare, use_container_width=True)

                    

                    # Add overall score for comparison

                    raw_score = _calculate_score(raw_metrics)

                    debiased_score = _calculate_score(deb_metrics)

                    st.markdown(f"**Overall Score (Raw Dataset):** {raw_score:.2f}/100")

                    st.markdown(f"**Overall Score (Debiased Dataset):** {debiased_score:.2f}/100")



                    try:

                        labels = list(raw_metrics.keys())

                        raw_vals = [raw_metrics[k] for k in labels]

                        deb_vals = [deb_metrics.get(k, None) for k in labels]

                        fig, ax = plt.subplots(figsize=(7, 4))

                        x = np.arange(len(labels))

                        ax.bar(x - 0.2, raw_vals, width=0.4, label="Raw")

                        ax.bar(x + 0.2, deb_vals, width=0.4, label="Debiased")

                        ax.set_xticks(x)

                        ax.set_xticklabels(labels, rotation=15, ha="right")

                        ax.set_ylabel("Score")

                        ax.set_title("Before vs After Debiasing")

                        ax.legend()

                        st.pyplot(fig)

                    except Exception as e:

                        st.warning(f"Plotting failed: {e}")

                    

                    if detected == "Classification" and sensitive and sensitive != "None":

                        st.write("---")

                        st.subheader("Additional Visualizations (Classification)")



                        # Fairness-Performance Trade-off Plot

                        try:

                            # Re-run a quick audit on the test sets to get fairness metrics

                            raw_audit_metrics = audit.run_fairness_audit(raw_ytest, raw_preds, raw_df.loc[raw_ytest.index, sensitive])

                            deb_audit_metrics = audit.run_fairness_audit(deb_ytest, deb_preds, deb_df.loc[deb_ytest.index, sensitive])

                            

                            fairness_metric = abs(raw_audit_metrics.get("Demographic Parity Difference", 0.0))

                            deb_fairness_metric = abs(deb_audit_metrics.get("Demographic Parity Difference", 0.0))

                            

                            performance_metric = raw_metrics.get("F1", 0.0)

                            deb_performance_metric = deb_metrics.get("F1", 0.0)



                            fig, ax = plt.subplots(figsize=(7, 4))

                            ax.plot(fairness_metric, performance_metric, 'o', color='blue', markersize=10, label='Baseline (Raw)')

                            ax.plot(deb_fairness_metric, deb_performance_metric, 'o', color='green', markersize=10, label='Debiased')

                            ax.set_xlabel("Bias (Demographic Parity Difference)")

                            ax.set_ylabel("Performance (F1-score)")

                            ax.set_title("Fairness-Performance Trade-off")

                            ax.legend()

                            ax.set_xlim(left=0)

                            ax.set_ylim(bottom=0)

                            st.pyplot(fig)

                            st.info("The ideal model would be in the top-left corner (High Performance, Low Bias).")

                        except Exception as e:

                            st.warning(f"Could not generate trade-off plot. Ensure a sensitive attribute is selected in the Fairness Audit tab. Error: {e}")

                        

                        # Group-wise Performance Bar Chart

                        try:

                            st.write("#### F1-score by Sensitive Group")

                            if audit:

                                # Get F1-score for each group

                                def get_f1_by_group(y_true, y_pred, sensitive_attr):

                                    group_f1 = {}

                                    for group in sensitive_attr.unique():

                                        idx = sensitive_attr[sensitive_attr == group].index

                                        if len(idx) > 0:

                                            group_f1[group] = f1_score(y_true.loc[idx], y_pred.loc[idx], average="weighted", zero_division=0)

                                    return group_f1

                                raw_f1_groups = get_f1_by_group(raw_ytest, pd.Series(raw_preds, index=raw_ytest.index), raw_df.loc[raw_ytest.index, sensitive])

                                deb_f1_groups = get_f1_by_group(deb_ytest, pd.Series(deb_preds, index=deb_ytest.index), deb_df.loc[deb_ytest.index, sensitive])

                                group_f1_df = pd.DataFrame({

                                    "Raw Model F1": raw_f1_groups,

                                    "Debiased Model F1": deb_f1_groups

                                }).fillna(0).T

                                st.bar_chart(group_f1_df)

                                st.markdown("A fairer model will have similar F1 scores across all groups.")

                        except Exception as e:

                            st.warning(f"Could not generate group-wise F1-score plot. Error: {e}")



                        # Confusion Matrix Heatmaps

                        try:

                            st.write("#### Confusion Matrices")

                            col_raw, col_deb = st.columns(2)



                            with col_raw:

                                st.write("##### Raw Model")

                                cm_raw = confusion_matrix(raw_ytest, raw_preds)

                                fig, ax = plt.subplots(figsize=(4, 4))

                                sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax)

                                ax.set_title("Raw Model Confusion Matrix")

                                ax.set_xlabel("Predicted Label")

                                ax.set_ylabel("True Label")

                                st.pyplot(fig)



                            with col_deb:

                                st.write("##### Debiased Model")

                                cm_deb = confusion_matrix(deb_ytest, deb_preds)

                                fig, ax = plt.subplots(figsize=(4, 4))

                                sns.heatmap(cm_deb, annot=True, fmt='d', cmap='Greens', ax=ax)

                                ax.set_title("Debiased Model Confusion Matrix")

                                ax.set_xlabel("Predicted Label")

                                ax.set_ylabel("True Label")

                                st.pyplot(fig)

                        except Exception as e:

                            st.warning(f"Could not generate confusion matrices. Error: {e}")



                else:

                    st.info("No debiased dataset found. Run Debiasing first or upload debiased CSV.")



                # Model download (for baseline model)

                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tf:

                    pickle.dump(raw_model, tf)

                    tf_path = tf.name

                with open(tf_path, "rb") as f:

                    st.download_button("Download trained model (.pkl)", data=f, file_name="trained_model.pkl", mime="application/octet-stream")



            except Exception as e:

                st.error(f"Training failed: {e}")



    end_card()





# ---------------------------

# Simulator page

# ---------------------------

if st.session_state.page == "Simulator":

    start_card()

    st.markdown("<div class='section-title'>Bias Simulator</div>", unsafe_allow_html=True)

    if df is None:

        st.warning("Upload dataset first.")

    else:

        st.write("Adjust bias strength and simulate predictions.")

        bias_strength = st.slider("Bias strength", -1.0, 1.0, 0.0, 0.05)

        if simulator is None:

            st.info("Simulator backend missing.")

        else:

            target_guess = st.session_state.get("deb_target") or (df.columns[-1] if len(df.columns) else None)

            sens_guess = st.session_state.get("deb_sensitive") or (detect_sensitive_attributes(df)[0] if detect_sensitive_attributes(df) else (df.columns[0] if len(df.columns) else None))

            try:

                preds_sim, metrics_sim = simulator.simulate_bias_effect(

                    df,

                    target_guess if target_guess else df.columns[-1],

                    sens_guess if sens_guess else df.columns[0],

                    bias_strength

                )

                st.write(metrics_sim)

            except Exception as e:

                st.error(f"Simulator failed: {e}")

    end_card()



# ---------------------------

# Drift page

# ---------------------------

if st.session_state.page == "Drift":

    start_card()

    st.markdown("<div class='section-title'>Data Drift</div>", unsafe_allow_html=True)

    if df is None:

        st.warning("Upload dataset first.")

    else:

        prev_score = st.number_input("Previous model score", value=0.95, step=0.01)

        curr_score = st.number_input("Current model score", value=0.90, step=0.01)

        if st.button("Check Drift"):

            if drift is None:

                st.error("Drift backend missing.")

            else:

                try:

                    drifted = drift.detect_drift(prev_score, curr_score)

                    st.write("Drift detected:", drifted)

                except Exception as e:

                    st.error(f"Drift check failed: {e}")

    end_card()



# ---------------------------

# Community page (UPDATED WITH VOTING)

# ---------------------------

if st.session_state.page == "Community":

    start_card()

    st.markdown("<div class='section-title'>Community / Crowdsourcing</div>", unsafe_allow_html=True)



    if community is None:

        st.error("Community backend missing. Please ensure `backend/community.py` is available.")

    else:

        # --- Define callback for voting to update backend and session state ---

        def handle_vote(submission_id: str, vote_type: str):

            """Callback function to handle voting."""

            if vote_type == 'up':

                community.upvote_submission(submission_id)

            elif vote_type == 'down':

                community.downvote_submission(submission_id)

            st.session_state.voted_submissions.add(submission_id)



        # --- Initialize session state for tracking votes ---

        if 'voted_submissions' not in st.session_state:

            st.session_state.voted_submissions = set()



        st.write("### Community Submissions")

        st.caption("Vote on submissions that you find helpful or important.")



        try:

            subs = community.list_submissions()



            if subs.empty:

                st.info("No submissions yet. Be the first to contribute!")

            else:

                # --- Prepare DataFrame for display ---

                for col in ['upvotes', 'downvotes', 'submission_id']:

                    if col not in subs.columns:

                        if col == 'submission_id':

                            subs[col] = [f"fallback_{i}" for i in range(len(subs))]

                        else:

                            subs[col] = 0

                

                subs['score'] = subs['upvotes'].astype(int) - subs['downvotes'].astype(int)

                subs = subs.sort_values('score', ascending=False).reset_index(drop=True)



                # --- Display each submission with voting buttons ---

                for index, row in subs.iterrows():

                    sub_id = row['submission_id']

                    has_voted = sub_id in st.session_state.voted_submissions



                    with st.container():

                        st.markdown("---")

                        vote_col, content_col = st.columns([0.15, 0.85])



                        with vote_col:

                            st.button("▲", key=f"up_{sub_id}", on_click=handle_vote, args=(sub_id, 'up'), disabled=has_voted, use_container_width=True)

                            st.markdown(f"<h4 style='text-align: center; color: #2D58A0;'>{row['score']}</h4>", unsafe_allow_html=True)

                            st.button("▼", key=f"down_{sub_id}", on_click=handle_vote, args=(sub_id, 'down'), disabled=has_voted, use_container_width=True)



                        with content_col:

                            st.markdown(f"**{row.get('submission_type', 'Submission').capitalize()}** by *{row.get('name', 'Anonymous')}*")

                            timestamp_str = pd.to_datetime(row.get('timestamp')).strftime('%b %d, %Y %H:%M')

                            st.caption(f"ID: `{sub_id}` | Posted on {timestamp_str}")

                            st.write(row.get('content', '*No content provided.*'))

                            if row.get('attached_filename') and row.get('attached_filename') not in [None, ""]:

                                st.info(f"Attachment included: `{row.get('attached_filename')}`")

        except Exception as e:

            st.error(f"Could not load community submissions. The backend might need an update. Error: {e}")



        # --- Submission Form ---

        st.markdown("---")

        st.write("### Submit to the community")

        with st.form("submission_form", clear_on_submit=True):

            name = st.text_input("Your name", placeholder="Anonymous")

            email = st.text_input("Your email (optional)")

            role = st.text_input("Your role / organization (optional)")

            sub_type = st.selectbox("Submission type", ["Report", "Dataset", "Comment", "Suggestion"])

            content = st.text_area("Describe your submission (what, why, notes)", height=150)

            attached = st.file_uploader("Attach a file (optional)", type=["csv", "xlsx", "json", "txt", "pdf"])

            

            submitted = st.form_submit_button("Submit to Community")



            if submitted:

                attached_filename = None

                if attached is not None:

                    attach_dir = Path(BACKEND_PATH) / "data"

                    attach_dir.mkdir(parents=True, exist_ok=True)

                    attached_filename = f"{int(datetime.utcnow().timestamp())}_{attached.name}"

                    with open(attach_dir / attached_filename, "wb") as f:

                        f.write(attached.getbuffer())

                try:

                    community.add_submission(

                        name=name or "Anonymous",

                        email=email or "",

                        role=role or "",

                        submission_type=sub_type,

                        content=content or "",

                        attached_filename=attached_filename

                    )

                    st.success("Submission saved. Thank you! Refresh to see your post.")

                except Exception as e:

                    st.error(f"Failed to save submission: {e}")

    end_card()



# ---------------------------

# End of app

# ---------------------------