"""
Enhanced Data Analysis Assistant v3
Single-file Streamlit app with:
- Upload / multiple file support
- Advanced cleaning (missing, dtypes, outliers)
- Interactive EDA (Plotly)
- AutoML comparison (PyCaret optional)
- Model training (sklearn) + export
- SHAP explainability (optional)
- AI Chat helper (OpenAI API integration - requires key)
- PDF export (FPDF) of report
- Memory & history
- Simple auth, caching, performance improvements

Notes:
- Some features are optional and will show friendly messages if packages are missing.
- Set OPENAI_API_KEY as an environment variable for chat features.
- Install dependencies: streamlit, pandas, numpy, plotly, scikit-learn, shap, pycaret, fpdf, joblib
  (pycaret can be heavy; app will still run without it.)

Run: streamlit run enhanced_data_analysis_assistant_v3.py
"""

import os
import io
import base64
import time
import tempfile
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Optional imports with graceful degradation
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    import shap
except Exception:
    shap = None

try:
    from pycaret.classification import setup as pyc_setup, compare_models, pull as pyc_pull
    from pycaret.regression import setup as pyr_setup
    HAS_PYCARET = True
except Exception:
    HAS_PYCARET = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import joblib
except Exception as e:
    st.warning(f"Some sklearn components missing: {e}")

try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ---------------------- Helpers ----------------------

st.set_page_config(page_title="Enhanced Data Analysis Assistant v3", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = {}
if "df" not in st.session_state:
    st.session_state.df = None
if "models" not in st.session_state:
    st.session_state.models = {}


def log(action):
    entry = {"time": datetime.now().isoformat(), "action": action}
    st.session_state.history.append(entry)


def file_uploader_multifile():
    uploaded_files = st.file_uploader("Upload CSV/Excel files", accept_multiple_files=True)
    if uploaded_files:
        dfs = []
        for f in uploaded_files:
            try:
                if f.name.endswith(".csv"):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
                dfs.append(df)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
        if len(dfs) == 1:
            return dfs[0]
        elif len(dfs) > 1:
            st.info("Merging multiple files by vertical concat. If you need different merge logic, handle externally.")
            return pd.concat(dfs, ignore_index=True)
    return None


@st.cache_data
def infer_dtypes(df):
    return df.dtypes.apply(lambda x: str(x)).to_dict()


def quick_clean(df):
    df = df.copy()
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # auto parse dates
    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c])
                df[c] = parsed
            except Exception:
                pass
    return df


def impute_and_encode(df, numeric_strategy="median"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy=numeric_strategy)), ("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preprocessor = ColumnTransformer([("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)])
    return preprocessor, num_cols, cat_cols


def detect_outliers(df, col, method="iqr", factor=1.5):
    if method == "iqr":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return df[(df[col] < lower) | (df[col] > upper)]
    return pd.DataFrame()


def plot_interactive(df):
    if px is None:
        st.warning("Plotly not available. Install plotly for interactive visuals.")
        return
    st.subheader("Interactive Visualizations")
    cols = st.multiselect("Choose columns for scatter (x,y,color)", df.columns.tolist(), default=df.columns[:3].tolist())
    if len(cols) >= 2:
        x = cols[0]
        y = cols[1]
        color = cols[2] if len(cols) >= 3 else None
        fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {x} vs {y}")
        st.plotly_chart(fig, use_container_width=True)


def run_shap_explainer(model, X):
    if shap is None:
        st.error("SHAP is not installed. Install shap to use explainability features.")
        return
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader("SHAP summary")
    try:
        shap.plots.beeswarm(shap_values)
        st.pyplot(bbox_inches='tight')
    except Exception:
        st.write(shap_values)


def save_model(model, name="model.pkl"):
    joblib.dump(model, name)
    st.success(f"Model saved to {name}")
    return name


def generate_pdf_report(title, text_blocks, image_bytes_list=None, out_path="report.pdf"):
    if not HAS_FPDF:
        st.error("FPDF not installed. Install fpdf to enable PDF export.")
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
            tmp.write(b)
            tmp.close()
            pdf.image(tmp.name, x=10, y=20, w=180)
    pdf.output(out_path)
    return out_path


def openai_chat(prompt):
    if not HAS_OPENAI:
        return "OpenAI package not installed. Install openai to use chat."
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "Set environment variable OPENAI_API_KEY to enable AI chat."
    openai.api_key = key
    try:
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"


# ---------------------- UI ----------------------

st.sidebar.title("🧠 Enhanced Assistant v3")
mode = st.sidebar.radio("Mode", ["Main App", "AI Chat", "Settings"]) 

# Simple auth (demo)
with st.sidebar.expander("🔐 Login (demo)"):
    password = st.text_input("Enter app password", type="password")
    if password:
        if password == "prem":
            st.sidebar.success("Authenticated")
        else:
            st.sidebar.error("Wrong password (demo: 'prem')")

if mode == "Settings":
    st.header("Settings & Integrations")
    st.markdown("- Set OPENAI_API_KEY as environment variable to enable AI Chat features.")
    st.markdown(f"PyCaret available: {HAS_PYCARET}")
    st.markdown(f"SHAP available: {shap is not None}")
    st.markdown(f"FPDF available: {HAS_FPDF}")
    st.markdown("Install missing packages with pip if you want full functionality.")

elif mode == "AI Chat":
    st.header("AI Assistant — Ask about your dataset or results")
    user_prompt = st.text_area("Ask anything (data summary, model suggestions, explain charts)")
    if st.button("Ask AI"):
        if user_prompt.strip() == "":
            st.warning("Enter a question first.")
        else:
            with st.spinner("AI is thinking..."):
                ans = openai_chat(user_prompt)
                st.markdown("**AI Response:**")
                st.write(ans)
                log(f"AI question: {user_prompt}")

else:
    st.title("Enhanced Data Analysis Assistant")
    st.markdown("Upload data, run EDA, train models, explain with SHAP, and export reports.")

    tabs = st.tabs(["Upload", "Cleaning", "EDA", "Modeling", "Explain", "Export", "Memory & History"])

    # ---------------- Upload Tab ----------------
    with tabs[0]:
        st.header("Upload Data")
        df = file_uploader_multifile()
        if df is not None:
            st.session_state.df = df
            st.success("Data loaded into session")
            st.dataframe(df.head())
            log("Uploaded data")
        if st.session_state.df is not None:
            if st.button("Preview full summary"):
                st.write(st.session_state.df.describe(include='all'))

    # ---------------- Cleaning Tab ----------------
    with tabs[1]:
        st.header("Data Cleaning & Preprocessing")
        if st.session_state.df is None:
            st.warning("Upload data first")
        else:
            df = st.session_state.df
            st.subheader("Quick Info")
            st.write(pd.DataFrame.from_dict(infer_dtypes(df), orient='index', columns=['dtype']))
            if st.button("Run Quick Clean (strip cols, parse dates)"):
                st.session_state.df = quick_clean(df)
                st.success("Quick clean applied")
                log("Quick clean")
            st.markdown("---")
            col = st.selectbox("Select numeric column to detect outliers (IQR)", df.select_dtypes(include=[np.number]).columns.tolist() if not df.empty else [])
            if col:
                out = detect_outliers(df, col)
                st.write(f"Detected {len(out)} outliers")
                if st.button("Remove outliers"):
                    st.session_state.df = df[~df.index.isin(out.index)].reset_index(drop=True)
                    st.success("Outliers removed")
                    log(f"Removed outliers in {col}")

    # ---------------- EDA Tab ----------------
    with tabs[2]:
        st.header("Exploratory Data Analysis")
        if st.session_state.df is None:
            st.warning("Upload data first")
        else:
            df = st.session_state.df
            st.subheader("Basic stats")
            st.write(df.describe(include='all').T)
            st.markdown("---")
            if px is not None:
                plot_interactive(df)
            else:
                st.info("Install plotly for interactive charts")
            st.markdown("---")
            if st.button("Show Correlation Heatmap"):
                try:
                    num = df.select_dtypes(include=[np.number])
                    corr = num.corr()
                    if px is not None:
                        fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write(corr)
                except Exception as e:
                    st.error(e)

    # ---------------- Modeling Tab ----------------
    with tabs[3]:
        st.header("Modeling")
        if st.session_state.df is None:
            st.warning("Upload data first")
        else:
            df = st.session_state.df
            task = st.selectbox("Task", ["Classification", "Regression"])
            target = st.selectbox("Select target column", df.columns.tolist())
            test_size = st.slider("Test size", 0.1, 0.5, 0.2)
            if st.button("Train baseline model"):
                X = df.drop(columns=[target])
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                # Simple pipeline: numeric impute + onehot, then RF
                pre, num_cols, cat_cols = impute_and_encode(df.drop(columns=[target]))
                try:
                    # Fit preprocessor on train
                    pre.fit(X_train)
                    X_train_trans = pre.transform(X_train)
                    X_test_trans = pre.transform(X_test)
                except Exception:
                    # fallback: fill na
                    X_train_trans = X_train.fillna(0)
                    X_test_trans = X_test.fillna(0)
                if task == "Classification":
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train_trans, y_train)
                preds = model.predict(X_test_trans)
                if task == "Classification":
                    acc = accuracy_score(y_test, preds)
                    st.success(f"Baseline RandomForest accuracy: {acc:.4f}")
                    st.text(classification_report(y_test, preds))
                else:
                    mse = mean_squared_error(y_test, preds)
                    st.success(f"Baseline RandomForest MSE: {mse:.4f}")
                name = st.text_input("Model filename to save", "model_baseline.pkl")
                if st.button("Save Model"):
                    save_model(model, name)
                    st.session_state.models[name] = model
                    log(f"Saved model {name}")

            st.markdown("---")
            st.subheader("AutoML (PyCaret)")
            if not HAS_PYCARET:
                st.info("PyCaret not installed. Install pycaret to enable AutoML.")
            else:
                if st.button("Run AutoML Comparison (PyCaret)"):
                    try:
                        if task == "Classification":
                            s = pyc_setup(df, target=target, silent=True, session_id=123)
                            best = compare_models()
                            st.write(best)
                            st.dataframe(pyc_pull())
                            log("Ran PyCaret AutoML")
                        else:
                            s = pyr_setup(df, target=target, silent=True, session_id=123)
                            best = compare_models()
                            st.write(best)
                            st.dataframe(pyc_pull())
                            log("Ran PyCaret AutoML")
                    except Exception as e:
                        st.error(e)

    # ---------------- Explain Tab ----------------
    with tabs[4]:
        st.header("Explainability & Feature Importance")
        if st.session_state.models:
            model_name = st.selectbox("Choose a saved model", list(st.session_state.models.keys()))
            model = st.session_state.models.get(model_name)
            if model is None:
                st.warning("No model found")
            else:
                df = st.session_state.df
                X = df.drop(columns=[st.selectbox("Select target for SHAP (for building X)", df.columns.tolist())])
                st.write("Running SHAP (may be slow)")
                if shap is not None:
                    try:
                        explainer = shap.Explainer(model, X)
                        sv = explainer(X)
                        st.subheader("SHAP summary plot")
                        shap.plots.beeswarm(sv)
                        st.pyplot(bbox_inches='tight')
                    except Exception as e:
                        st.error(e)
                else:
                    st.info("Install SHAP to enable explainability")
        else:
            st.info("Save a model first to use explainability")

    # ---------------- Export Tab ----------------
    with tabs[5]:
        st.header("Export & Reporting")
        if st.session_state.df is None:
            st.warning("Upload data first")
        else:
            df = st.session_state.df
            if st.button("Download CSV"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "data_export.csv", "text/csv")
            st.markdown("---")
            st.subheader("Generate PDF Report")
            title = st.text_input("Report title", "Data Analysis Report")
            notes = st.text_area("Summary notes (will appear in PDF)", value="Add your summary here")
            if st.button("Generate PDF"):
                images = []
                # create a small plot for report
                if px is not None:
                    fig = px.histogram(df.select_dtypes(include=[np.number]).iloc[:, 0:1].dropna().melt(), x='value', title='Sample Distribution')
                    buf = io.BytesIO()
                    fig.write_image(buf, format='png')
                    images.append(buf.getvalue())
                out = generate_pdf_report(title, [notes, f"Generated on {datetime.now().isoformat()}"], image_bytes_list=images, out_path="report.pdf")
                if out:
                    with open(out, "rb") as f:
                        st.download_button("Download Report PDF", f, file_name="report.pdf")
                        log("Generated PDF report")

    # ---------------- Memory & History Tab ----------------
    with tabs[6]:
        st.header("Memory & History")
        st.subheader("Session History")
        st.write(pd.DataFrame(st.session_state.history))
        st.subheader("Memory (key -> value)")
        st.write(st.session_state.memory)
        with st.form("add_memory"):
            k = st.text_input("Key")
            v = st.text_area("Value")
            submitted = st.form_submit_button("Save Memory")
            if submitted and k:
                st.session_state.memory[k] = v
                st.success("Saved")
                log(f"Memory saved: {k}")

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("Tip: install optional packages (pycaret, shap, plotly, fpdf) for full functionality.")


# End of file
