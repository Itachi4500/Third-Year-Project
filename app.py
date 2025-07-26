import streamlit as st
import pandas as pd
from utils.cleaner import clean_data
from utils.eda import run_eda
from utils.visualizer import show_visuals
from utils.modeler import run_modeling
from utils.exporter import export_data
from utils.powerbi_pipeline import powerbi_pipeline
from utils.live_dashboard import live_dashboard


# ------------------------------
# 🌙 Theme Switcher
# ------------------------------
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"], index=0)

light_css = """
<style>
body {
    background-color: #f9f9fb;
    color: #000;
}
</style>
"""

dark_css = """
<style>
body {
    background-color: #1e1e2f;
    color: #f5f5f5;
}
.sidebar .sidebar-content {
    background-color: #2c2f3a;
    color: #f5f5f5;
}
</style>
"""

st.markdown(light_css if theme == "Light" else dark_css, unsafe_allow_html=True)

# ------------------------------
# 🎉 CSS: Sidebar + Animations
# ------------------------------
st.markdown("""
    <style>
    .main {
        animation: fadein 1s ease-in;
    }
    @keyframes fadein {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    .sidebar .sidebar-content {
        padding: 1rem;
        background-color: #f0f2f6;
        border-right: 2px solid #ddd;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# 🤖 Branding Header
# ------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Robot_icon.svg/240px-Robot_icon.svg.png", width=70)
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
if "cleaned_df" in st.session_state:
    st.session_state.df = st.session_state.cleaned_df

st.sidebar.markdown("---")
if st.session_state.df is not None:
    st.sidebar.success("✅ Dataset Loaded")
    st.sidebar.write(f"Rows: {st.session_state.df.shape[0]}")
    st.sidebar.write(f"Columns: {st.session_state.df.shape[1]}")
else:
    st.sidebar.error("🚫 No Dataset Loaded")

# ------------------------------
# 🧭 Main Navigation Logic
# ------------------------------
st.title(menu[choice])
st.markdown("---")

if choice == "📁 Upload Dataset":
    uploaded_file = st.file_uploader("📤 Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(st.session_state.df.head())

elif choice == "🧹 Data Cleaning":
    if st.session_state.df is not None:
        st.session_state.df = clean_data(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset first.")

elif choice == "📊 Exploratory Data Analysis":
    if st.session_state.df is not None:
        run_eda(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset first.")

elif choice == "📈 Visualization":
    if st.session_state.df is not None:
        show_visuals(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset first.")

elif choice == "🧠 Feature Engineering & Modeling":
    if st.session_state.df is not None:
        run_modeling(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset first.")

elif choice == "📌 Evaluation & Tuning":
    st.info("📌 Model evaluation & tuning is handled during modeling. See results after training.")

elif choice == "⬇️ Export Dataset":
    if st.session_state.df is not None:
        export_data(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset first.")
# Bower BI Pipeline
elif choice == "📊 Power BI Pipeline":
    if "cleaned_df" in st.session_state:
        df = st.session_state.cleaned_df
        powerbi_pipeline(df)
    else:
        st.warning("⚠️ Please upload and clean your dataset first.")
        
# Live Dashboard
elif choice == "📡 Live Dashboard":
    if "cleaned_df" in st.session_state:
        df = st.session_state.cleaned_df
        live_dashboard(df)
    else:
        st.warning("Please upload and clean your dataset first.")

# ------------------------------
# 📝 Auto-Generated Report Modal
# ------------------------------
elif choice == "📝 Generate Report":
    if st.session_state.df is not None:
        st.subheader("📄 Data Summary Report")

        with st.expander("📌 View Quick Stats"):
            st.write(st.session_state.df.describe(include="all").transpose())

        with st.expander("🧠 Feature Summary"):
            st.write("Total columns:", st.session_state.df.shape[1])
            st.write("Categorical columns:", list(st.session_state.df.select_dtypes(include='object').columns))
            st.write("Numeric columns:", list(st.session_state.df.select_dtypes(include=['int64', 'float64']).columns))

        with st.expander("🚨 Missing Values Summary"):
            st.write(st.session_state.df.isnull().sum().to_frame(name="Missing Values"))

        st.success("✅ Report Ready — You can export or take screenshots.")
    else:
        st.warning("⚠️ Upload and clean a dataset before generating the report.")
        

