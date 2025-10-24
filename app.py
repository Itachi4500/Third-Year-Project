# app_fast_v4.py - Enhanced Data Analysis Assistant v4 (speed-optimized for ~1M rows)
import os, io, tempfile, warnings
from datetime import datetime
import streamlit as st, pandas as pd, numpy as np
from utils.upload import upload_data
from utils.cleaner import clean_data
from utils.eda import run_eda
from utils.visualizer import show_visuals
from utils.modeler import run_modeling
from utils.exporter import export_data
from utils.memory import remember, recall, forget, clear_all_memory, show_memory, show_memory_history
from utils.powerbi_pipeline import powerbi_pipeline
from utils.refresh import refresh_data
warnings.filterwarnings("ignore")
st.set_page_config(page_title="🧠 Enhanced Data Analysis Assistant", layout="wide")
st.title("🧠 Enhanced Data Analysis Assistant")

# --- Session Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None

# --- Advanced Sidebar Navigation ---
nav = st.sidebar.radio("📌 Navigation", [
    "Refresh",
    "Upload Data",
    "Data Cleaning",
    "EDA",
    "Visualizations",
    "Model Training",
    "Power BI Pipeline",
    "Memory & Notes",
    "Export"
])

# --- Navigation Routing ---
if nav == "Upload Data":
    df = upload_data()
    if df is not None:
        st.session_state.df = df
        st.dataframe(df.head())

elif nav == "Refresh":
    refresh_data()

elif nav == "Data Cleaning":
    if st.session_state.df is not None:
        df_cleaned = clean_data(st.session_state.df)
        st.session_state.df = df_cleaned
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "EDA":
    if st.session_state.df is not None:
        run_eda(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Visualizations":
    if st.session_state.df is not None:
        show_visuals(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Model Training":
    if st.session_state.df is not None:
        run_modeling(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Power BI Pipeline":
    if st.session_state.df is not None:
        powerbi_pipeline(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")

elif nav == "Memory & Notes":
    show_memory()
    show_memory_history()
    with st.expander("➕ Add Note"):
        key = st.text_input("Memory Key (e.g. 'notes.data.cleaning')")
        value = st.text_area("Memory Value")
        if st.button("💾 Remember"):
            remember(key, value)
    with st.expander("❌ Forget Note"):
        forget_key = st.text_input("Key to forget")
        if st.button("Forget"):
            forget(forget_key)
    if st.button("🧹 Clear All Memory"):
        clear_all_memory()

elif nav == "Export":
    if st.session_state.df is not None:
        export_data(st.session_state.df)
    else:
        st.warning("📂 Please upload a dataset first.")


# Optional libs
try: import plotly.express as px
except: px = None
try: import shap
except: shap = None
try:
    from pycaret.classification import setup as pyc_setup, compare_models, pull as pyc_pull
    from pycaret.regression import setup as pyr_setup
    HAS_PYCARET = True
except:
    HAS_PYCARET = False
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
    import joblib
except Exception:
    st.warning("sklearn/joblib missing or partially available")

try: from fpdf import FPDF; HAS_FPDF = True
except: HAS_FPDF = False

try: import openai; HAS_OPENAI = True
except: HAS_OPENAI = False

# UI config
st.set_page_config("Enhanced Data Analysis Assistant v4", layout="wide")
st.title("🧠 Enhanced Data Analysis Assistant v4 — Fast (1M+ rows ready)")

# session state defaults
st.session_state.setdefault("df", None)
st.session_state.setdefault("models", {})
st.session_state.setdefault("history", [])
st.session_state.setdefault("memory", {})

def log(action):
    st.session_state.history.append({"time": datetime.now().isoformat(), "action": action})

# ---------------------- Fast I/O & dtype helpers ----------------------
@st.cache_data(show_spinner=False)
def infer_dtypes_fast(sample_df):
    d = {}
    for c, t in sample_df.dtypes.items():
        if pd.api.types.is_integer_dtype(t): d[c] = "int64"
        elif pd.api.types.is_float_dtype(t): d[c] = "float64"
        elif pd.api.types.is_bool_dtype(t): d[c] = "bool"
        elif pd.api.types.is_datetime64_any_dtype(t): d[c] = "datetime64"
        else: d[c] = "object"
    return d

def read_csv_chunked(uploaded_file, dtype=None, parse_dates=None, chunksize=200_000):
    it = pd.read_csv(uploaded_file, dtype=dtype, parse_dates=parse_dates, chunksize=chunksize, low_memory=False)
    chunks = []
    p = st.progress(0)
    total = 0
    for i, chunk in enumerate(it, start=1):
        chunks.append(chunk)
        total += len(chunk)
        p.progress(min(100, int(total / (chunksize * 5) * 100)))  # rough progress
    p.empty()
    return pd.concat(chunks, ignore_index=True)

def load_file(file_obj):
    name = getattr(file_obj, "name", "uploaded")
    if name.endswith(".csv"):
        # Try quick sampling to infer dtypes, then read chunked with inferred dtypes
        sample = pd.read_csv(file_obj, nrows=5000, low_memory=False)
        dtypes = infer_dtypes_fast(sample)
        # Reset file pointer then read chunked
        file_obj.seek(0)
        df = read_csv_chunked(file_obj, dtype={k: None for k in dtypes})
    else:
        # excel (smaller typically)
        df = pd.read_excel(file_obj)
    return df

# ---------------------- Fast preprocessing ----------------------
@st.cache_data
def convert_categoricals(df, max_unique_for_cat=2000):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 50):
            if df[c].nunique() <= max_unique_for_cat:
                df[c] = df[c].astype("category")
    return df

@st.cache_data
def fast_impute_and_encode(df, target_col=None, numeric_fill_method="median"):
    df = df.copy()
    # Split
    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None
    # Numerics: fillna with median (vectorized)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        if numeric_fill_method == "median":
            medians = X[num_cols].median()
            X[num_cols] = X[num_cols].fillna(medians)
        else:
            X[num_cols] = X[num_cols].fillna(0)
    # Cats: convert to category and use codes (fast and memory-sparing)
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        # keep -1 for NA
        codes = X[c].cat.codes
        codes = codes.replace(-1, np.nan).astype("Float64")
        X[c] = codes
        # numeric fill for codes
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    # Final fill for any remaining NA
    X = X.fillna(0)
    return X, y, num_cols, cat_cols

# ---------------------- Caching models ----------------------
@st.cache_resource
def train_rf(X_train, y_train, task="classification", n_estimators=100):
    if task == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model

# ---------------------- Utilities ----------------------
def sample_for_display(df, n=5000):
    if len(df) <= n: return df
    return df.sample(n, random_state=42)

def generate_pdf_report(title, text_blocks, image_bytes_list=None, out_path="report.pdf"):
    if not HAS_FPDF:
        st.error("FPDF not installed. Install fpdf for PDF export.")
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    for block in text_blocks:
        pdf.multi_cell(0, 8, block)
        pdf.ln(2)
    if image_bytes_list:
        for b in image_bytes_list:
            pdf.add_page()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(b); tmp.close()
            pdf.image(tmp.name, x=10, y=20, w=180)
    pdf.output(out_path)
    return out_path

def openai_chat(prompt):
    if not HAS_OPENAI: return "OpenAI package not installed."
    key = os.getenv("OPENAI_API_KEY")
    if not key: return "Set OPENAI_API_KEY environment variable."
    openai.api_key = key
    try:
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"

# ---------------------- Sidebar / settings ----------------------
st.sidebar.header("Settings & Speed Options")
sample_size = st.sidebar.number_input("Interactive plot / SHAP / AutoML sample size", min_value=500, max_value=100_000, value=5000, step=500)
chunksize = st.sidebar.number_input("CSV chunksize for upload", min_value=50_000, max_value=500_000, value=200_000, step=50_000)
rf_estimators = st.sidebar.slider("RandomForest n_estimators (baseline)", 10, 500, 100)
st.sidebar.caption("For huge datasets: increase chunksize, reduce sample_size, use n_jobs=-1 (enabled).")

mode = st.sidebar.radio("Mode", ["Main App", "AI Chat", "Settings"])
with st.sidebar.expander("🔐 Login (demo)"):
    pwd = st.text_input("Enter app password", type="password")
    if pwd:
        if pwd == "prem":
            st.sidebar.success("Authenticated")
        else:
            st.sidebar.error("Wrong password (demo: 'prem')")

if mode == "Settings":
    st.header("Integrations")
    st.write(f"PyCaret available: {HAS_PYCARET}")
    st.write(f"SHAP available: {shap is not None}")
    st.write(f"FPDF available: {HAS_FPDF}")
    st.write(f"OpenAI available: {HAS_OPENAI}")
    st.stop()

if mode == "AI Chat":
    st.header("AI Assistant — Ask about your dataset or results")
    user_prompt = st.text_area("Ask anything (dataset-aware answers fast)")
    if st.button("Ask AI"):
        st.write(openai_chat(user_prompt))
        log(f"AI question: {user_prompt}")
    st.stop()

# ---------------------- Main App ----------------------
st.header("Upload Dataset (CSV / Excel)")
uploaded_files = st.file_uploader("Upload CSV/Excel files (multiple allowed)", accept_multiple_files=True)
if uploaded_files:
    # handle multiple files - we do not blindly concat 1M+ files; show info and confirm
    if len(uploaded_files) > 1:
        st.info(f"{len(uploaded_files)} files uploaded — they will be concatenated vertically (ensure columns match).")
    with st.spinner("Reading files (fast mode)..."):
        dfs = []
        for f in uploaded_files:
            try:
                # for large CSV, use chunked loader
                if f.name.endswith(".csv"):
                    f.seek(0)
                    df_part = read_csv_chunked(f, chunksize=chunksize)
                else:
                    df_part = pd.read_excel(f)
                dfs.append(df_part)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
        if dfs:
            # concat but avoid copies where possible
            st.session_state.df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            st.success(f"Loaded {len(st.session_state.df):,} rows")
            log("Uploaded data")

if st.session_state.df is None:
    st.info("Upload data to begin (optimized for large files).")
    st.stop()

# quick peek & dtype conversion
df = st.session_state.df
st.subheader("Data preview (sampled)")
st.write(df.head(3))
if st.button("Convert likely object columns to category (fast)"):
    df = convert_categoricals(df)
    st.session_state.df = df
    st.success("Converted object columns to category where appropriate.")
    log("Converted categories")

# Quick summary (fast)
st.subheader("Fast summary")
num = df.select_dtypes(include=[np.number]).shape[1]
cat = df.select_dtypes(include=["category", "object"]).shape[1]
st.write(f"Rows: {len(df):,} | Numeric cols: {num} | Categorical cols: {cat}")
if st.button("Show full describe (may be slow)"):
    st.write(df.describe(include='all').T)

# ---------------------- Cleaning utilities ----------------------
st.subheader("Cleaning & Preprocessing")
if st.button("Quick parse datetimes (vectorized)"):
    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                # only replace if many values parsed
                if parsed.notna().sum() / max(1, len(parsed)) > 0.6:
                    df[c] = parsed
            except Exception:
                pass
    st.session_state.df = df
    st.success("Datetime parsing attempted (vectorized).")
    log("Parsed datetimes")

# Outlier detection (fast, vectorized)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
col = st.selectbox("Select numeric column to detect outliers (IQR)", numeric_cols)
if col:
    factor = st.number_input("IQR factor", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
    q1 = float(df[col].quantile(0.25)); q3 = float(df[col].quantile(0.75)); iqr = q3 - q1
    lower = q1 - factor * iqr; upper = q3 + factor * iqr
    out_mask = (df[col] < lower) | (df[col] > upper)
    st.write(f"Outliers detected: {out_mask.sum():,}")
    if st.button("Remove outliers (fast)"):
        df = df.loc[~out_mask].reset_index(drop=True)
        st.session_state.df = df
        st.success("Outliers removed")
        log(f"Removed outliers in {col}")

# ---------------------- EDA (sampled) ----------------------
st.subheader("Exploratory Data Analysis (sampled)")
if px is None:
    st.info("Install plotly for interactive charts.")
else:
    samp = sample_for_display(df, n=sample_size)
    cols = st.multiselect("Choose columns for scatter (x,y,color)", samp.columns.tolist(), default=samp.columns[:3].tolist())
    if len(cols) >= 2:
        fig = px.scatter(samp, x=cols[0], y=cols[1], color=(cols[2] if len(cols) >= 3 else None), title=f"Scatter: {cols[0]} vs {cols[1]}")
        st.plotly_chart(fig, use_container_width=True)
if st.button("Show correlation heatmap (sample)"):
    if px:
        s = sample_for_display(df.select_dtypes(include=[np.number]), n=sample_size)
        if s.shape[1] > 0:
            st.plotly_chart(px.imshow(s.corr(), text_auto=True, title="Correlation (sample)"), use_container_width=True)
        else:
            st.info("No numeric columns for correlation.")

# ---------------------- Modeling ----------------------
st.subheader("Modeling (fast baseline + optional AutoML)")
task = st.selectbox("Task", ["Classification", "Regression"])
target = st.selectbox("Select target column", df.columns.tolist())
test_size = st.slider("Test size", 0.05, 0.5, 0.2)

if st.button("Train fast baseline model"):
    with st.spinner("Preprocessing (fast vectorized)..."):
        X_all, y_all, num_cols, cat_cols = fast_impute_and_encode(df, target_col=target)
    # sample for training if dataset very large to reduce model time, but user asked for speed - we do full if feasible
    use_full = st.checkbox("Train on full dataset (may be slow) — otherwise sample", value=False)
    if not use_full and len(X_all) > 200_000:
        X_train_full = X_all.sample(200_000, random_state=42)
        y_train_full = y_all.loc[X_train_full.index]
        st.info("Using sampled 200k rows for speed.")
    else:
        X_train_full, y_train_full = X_all, y_all

    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=42)
    st.info("Training RandomForest (n_jobs=-1)...")
    model = train_rf(X_train, y_train, task=("classification" if task == "Classification" else "regression"), n_estimators=rf_estimators)
    preds = model.predict(X_test)
    if task == "Classification":
        acc = accuracy_score(y_test, preds)
        st.success(f"RandomForest accuracy (baseline): {acc:.4f}")
        st.text(classification_report(y_test, preds, zero_division=0))
    else:
        mse = mean_squared_error(y_test, preds)
        st.success(f"RandomForest MSE (baseline): {mse:.4f}")
    # Save model
    name = st.text_input("Model filename", value=f"rf_baseline_{int(time:=datetime.now().timestamp())}.pkl")
    if st.button("Save model to disk"):
        joblib.dump(model, name)
        st.session_state.models[name] = model
        st.success(f"Saved {name}")
        log(f"Saved model {name}")

# PyCaret AutoML (sampled to be fast)
if HAS_PYCARET:
    if st.button("Run PyCaret AutoML (sampled)"):
        with st.spinner("Preparing sample for PyCaret..."):
            sample_for_automl = df.sample(min(len(df), max(10_000, sample_size)), random_state=42)
        try:
            if task == "Classification":
                s = pyc_setup(sample_for_automl, target=target, silent=True, html=False, session_id=123)
            else:
                s = pyr_setup(sample_for_automl, target=target, silent=True, html=False, session_id=123)
            best = compare_models()
            st.write(best)
            st.dataframe(pyc_pull())
            log("Ran PyCaret AutoML (sampled)")
        except Exception as e:
            st.error(f"PyCaret error: {e}")

# ---------------------- Explainability (SHAP sampled) ----------------------
st.subheader("Explainability (SHAP - sampled)")
if st.session_state.models:
    model_name = st.selectbox("Choose saved model", list(st.session_state.models.keys()))
    chosen_model = st.session_state.models.get(model_name)
else:
    chosen_model = None

if chosen_model is None and st.session_state.get("models"):
    # fallback choose any
    chosen_model = list(st.session_state.models.values())[0]

if chosen_model:
    if shap is None:
        st.info("Install SHAP for explainability.")
    else:
        with st.spinner("Preparing SHAP (sampled)..."):
            X_for_shap, _, _, _ = fast_impute_and_encode(df.drop(columns=[target]) if target in df.columns else df, target_col=None)
            Xs = sample_for_display(X_for_shap, n=sample_size)
            try:
                explainer = shap.TreeExplainer(chosen_model)
                shap_values = explainer.shap_values(Xs) if hasattr(explainer, "shap_values") else explainer(Xs)
                st.subheader("SHAP summary (sample)")
                try:
                    shap.plots.beeswarm(shap_values)
                    st.pyplot(bbox_inches="tight")
                except Exception:
                    st.write("SHAP produced values; plotting fallback.")
                    st.write(shap_values)
            except Exception as e:
                st.error(f"SHAP error: {e}")

else:
    st.info("Save or train a model first to use SHAP.")

# ---------------------- Export & Reporting ----------------------
st.subheader("Export & Reporting")
if st.button("Download sampled CSV (fast)"):
    samp = sample_for_display(df, n=sample_size)
    csv = samp.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (sample)", csv, "data_sample.csv", "text/csv")

st.markdown("---")
st.subheader("Generate PDF Report (fast)")
title = st.text_input("Report title", "Data Analysis Report")
notes = st.text_area("Summary notes", value="Add your summary here")
if st.button("Generate PDF (sample plots)"):
    images = []
    if px:
        try:
            s = sample_for_display(df.select_dtypes(include=[np.number]).iloc[:, :1].dropna(), n=sample_size)
            fig = px.histogram(s.melt(), x="value", title="Sample numeric distribution")
            buf = io.BytesIO()
            # fig.write_image requires kaleido; if not installed, we fallback to no image
            try:
                fig.write_image(buf, format="png")
                images.append(buf.getvalue())
            except Exception:
                st.warning("kaleido not available; PDF will exclude charts.")
        except Exception:
            pass
    out = generate_pdf_report(title, [notes, f"Generated on {datetime.now().isoformat()}"], image_bytes_list=images, out_path="report.pdf")
    if out:
        with open(out, "rb") as f:
            st.download_button("Download Report PDF", f, file_name="report.pdf")
            log("Generated PDF report")

# ---------------------- Memory & History ----------------------
st.subheader("Memory & History")
st.write(pd.DataFrame(st.session_state.history))
st.write(st.session_state.memory)
with st.form("add_memory"):
    k = st.text_input("Key")
    v = st.text_area("Value")
    submitted = st.form_submit_button("Save Memory")
    if submitted and k:
        st.session_state.memory[k] = v
        st.success("Saved")
        log(f"Memory saved: {k}")

st.caption("Tips: reduce sample_size for faster plots/SHAP/AutoML; set chunksize higher for faster file read. Use n_jobs=-1 for model training.")
