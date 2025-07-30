import streamlit as st
import pandas as pd
from utils.cleaner import clean_data
from utils.eda import run_eda
from utils.visualizer import show_visuals
from utils.modeler import run_modeling
from utils.exporter import export_data
from utils.powerbi_pipeline import powerbi_pipeline
from utils.live_dashboard import live_dashboard
from utils.refresh import refresh_data
from utils.memory import remember, recall, show_memory, clear_all_memory
from utils.layout import select_dashboard_size
from PIL import Image
import requests

# ------------------------------
# 🌙 Theme Switcher
# ------------------------------
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"], index=0)

base_css = """
<style>
body {
    background-color: %s;
    color: %s;
}
.sidebar .sidebar-content {
    background-color: %s;
    color: %s;
}
.main {
    animation: fadein 1s ease-in;
}
@keyframes fadein {
    0%% {opacity: 0;}
    100%% {opacity: 1;}
}
.block-container {
    padding-top: 2rem;
}
</style>
"""

if theme == "Light":
    st.markdown(base_css % ("#f9f9fb", "#000", "#f0f2f6", "#000"), unsafe_allow_html=True)
else:
    st.markdown(base_css % ("#1e1e2f", "#f5f5f5", "#2c2f3a", "#f5f5f5"), unsafe_allow_html=True)

# ------------------------------
# 🤖 Branding Header
# ------------------------------
@st.cache_resource
def load_logo():
    return Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Robot_icon.svg/240px-Robot_icon.svg.png", stream=True).raw)

st.sidebar.image(load_logo(), width=70)
st.sidebar.markdown("## 🤖 <b>Data Scientist Assistant</b>", unsafe_allow_html=True)

# ------------------------------
# Navigation Menu
# ------------------------------
menu = {
    "📁 Upload Dataset": "Upload Dataset",
    "🧹 Data Cleaning": "Data Cleaning",
    "📊 Exploratory Data Analysis": "Exploratory Data Analysis",
    "📈 Visualization": "Visualization",
    "🧠 Feature Engineering & Modeling": "Feature Engineering & Modeling",
    "📌 Evaluation & Tuning": "Evaluation & Tuning",
    "⬇️ Export Dataset": "Export Dataset",
    "📝 Generate Report": "Generate Report",
    "📊 Power BI Pipeline": "Power BI",
    "📡 Live Dashboard": "Dashboard"
}
choice = st.sidebar.radio("📂 **Select Operation**", list(menu.keys()))

# ------------------------------
# Dataset State
# ------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

st.sidebar.markdown("---")
if st.session_state.df is not None:
    st.sidebar.success("✅ Dataset Loaded")
    st.sidebar.write(f"Rows: {st.session_state.df.shape[0]}")
    st.sidebar.write(f"Columns: {st.session_state.df.shape[1]}")
else:
    st.sidebar.error("🚫 No Dataset Loaded")

# ------------------------------
# Main App Logic
# ------------------------------
st.title(menu[choice])
if choice == "📁 Upload Dataset":
    st.caption("Upload your CSV or Excel data file to begin the analysis process.")
elif choice == "🧹 Data Cleaning":
    st.caption("Clean your dataset by removing missing values and normalizing columns.")
elif choice == "📊 Exploratory Data Analysis":
    st.caption("Get insights into data distribution, types, and key statistics.")
elif choice == "📈 Visualization":
    st.caption("Create interactive visualizations for better understanding of your data.")
elif choice == "🧠 Feature Engineering & Modeling":
    st.caption("Apply machine learning models and feature engineering automatically.")
elif choice == "📌 Evaluation & Tuning":
    st.caption("Review model performance, accuracy, and key evaluation metrics.")
elif choice == "⬇️ Export Dataset":
    st.caption("Download the cleaned and transformed dataset for future use.")
elif choice == "📝 Generate Report":
    st.caption("Automatically generate summaries, stats, and missing value reports.")
elif choice == "📊 Power BI Pipeline":
    st.caption("Send cleaned data to Power BI for real-time visual dashboards.")
elif choice == "📡 Live Dashboard":
    st.caption("Explore live dashboards and monitor metrics interactively.")
st.markdown("---")

# Upload Dataset
if choice == "📁 Upload Dataset":
    uploaded_file = st.file_uploader("📤 Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

    @st.cache_data
    def load_data(uploaded_file):
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)

    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(st.session_state.df.head())

# Data Cleaning
elif choice == "🧹 Data Cleaning" and st.session_state.df is not None:
    df = st.session_state.df.copy()
    cleaning_steps = []

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧽 Drop Missing Values"):
            df.dropna(inplace=True)
            cleaning_steps.append("Dropped missing values")
            st.success("Missing values removed!")

    with col2:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cols_to_scale = st.multiselect("🔄 Normalize columns", numeric_cols)
        if st.button("⚖️ Normalize Selected Columns"):
            for col in cols_to_scale:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
            cleaning_steps.append(f"Normalized: {', '.join(cols_to_scale)}")
            st.success("Normalization complete.")

    if cleaning_steps:
        st.session_state.cleaned_df = df
        remember("cleaning_steps", cleaning_steps)

    st.markdown("### 🧠 Cleaning Memory")
    steps = recall("cleaning_steps")
    if steps:
        st.info(f"Steps: {steps}")
    show_memory()
    if st.button("🗑️ Clear Memory"):
        clear_all_memory()
        st.success("Memory cleared.")

# EDA
elif choice == "📊 Exploratory Data Analysis" and st.session_state.df is not None:
    run_eda(st.session_state.df)

# Visualization
elif choice == "📈 Visualization" and st.session_state.df is not None:
    show_visuals(st.session_state.df)

# Modeling
elif choice == "🧠 Feature Engineering & Modeling" and st.session_state.df is not None:
    run_modeling(st.session_state.df)

# Evaluation Placeholder
elif choice == "📌 Evaluation & Tuning":
    st.info("📌 Evaluation is integrated in modeling step.")

# Export
elif choice == "⬇️ Export Dataset" and st.session_state.df is not None:
    export_data(st.session_state.df)

# Power BI
elif choice == "📊 Power BI Pipeline" and "cleaned_df" in st.session_state:
    powerbi_pipeline(st.session_state.cleaned_df)

# Live Dashboard
elif choice == "📡 Live Dashboard" and "cleaned_df" in st.session_state:
    live_dashboard(st.session_state.cleaned_df)

# Report
elif choice == "📝 Generate Report" and st.session_state.df is not None:
    st.subheader("📄 Data Summary Report")

    with st.expander("📌 View Quick Stats"):
        st.write(st.session_state.df.describe(include="all").transpose())

    with st.expander("🧠 Feature Summary"):
        st.write("Total columns:", st.session_state.df.shape[1])
        st.write("Categorical:", list(st.session_state.df.select_dtypes(include='object').columns))
        st.write("Numeric:", list(st.session_state.df.select_dtypes(include=['int64', 'float64']).columns))

    with st.expander("🚨 Missing Values Summary"):
        st.write(st.session_state.df.isnull().sum().to_frame(name="Missing Values"))

    st.success("✅ Report Ready. You can export or take screenshots.")

else:
    st.warning("⚠️ Please upload a dataset to continue.")
