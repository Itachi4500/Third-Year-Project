import streamlit as st
from PIL import Image
import pandas as pd
from modules import upload, cleaning, eda, visualization, modeling, report, export, powerbi, dashboard

# Set page config
st.set_page_config(page_title="Data Scientist Assistant", page_icon="🧠", layout="wide")

# Apply theme
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"], index=0)
if theme != st.session_state.theme:
    st.session_state.theme = theme
    st.experimental_rerun()

# Load logo
@st.cache_data
def load_logo():
    return Image.open("static/logo.png")

st.sidebar.image(load_logo(), use_column_width=True)
st.sidebar.title("🧠 Data Scientist Assistant")

# Menu options
menu = ["📁 Upload Dataset", "🧹 Data Cleaning", "📊 Exploratory Data Analysis", "📈 Visualization",
         "🧠 Feature Engineering & Modeling", "📌 Evaluation & Tuning", "⬇️ Export Dataset",
         "📝 Generate Report", "📊 Power BI Pipeline", "📡 Live Dashboard"]

choice = st.sidebar.radio("Go to", menu)

# Show module descriptions
if choice == "📁 Upload Dataset":
    st.caption("Upload your CSV or Excel data file to begin the analysis process.")
    upload.render()
elif choice == "🧹 Data Cleaning":
    st.caption("Clean your dataset by removing missing values and normalizing columns.")
    cleaning.render()
elif choice == "📊 Exploratory Data Analysis":
    st.caption("Get insights into data distribution, types, and key statistics.")
    eda.render()
elif choice == "📈 Visualization":
    st.caption("Create interactive visualizations for better understanding of your data.")
    visualization.render()
elif choice == "🧠 Feature Engineering & Modeling":
    st.caption("Apply machine learning models and feature engineering automatically.")
    modeling.render()
elif choice == "📌 Evaluation & Tuning":
    st.caption("Review model performance, accuracy, and key evaluation metrics.")
    st.info("Evaluation is performed in the Modeling section using cross-validation.")
elif choice == "⬇️ Export Dataset":
    st.caption("Download the cleaned and transformed dataset for future use.")
    export.render()
elif choice == "📝 Generate Report":
    st.caption("Automatically generate summaries, stats, and missing value reports.")
    report.render()
elif choice == "📊 Power BI Pipeline":
    st.caption("Send cleaned data to Power BI for real-time visual dashboards.")
    powerbi.render()
elif choice == "📡 Live Dashboard":
    st.caption("Explore live dashboards and monitor metrics interactively.")
    dashboard.render()

# Add footer or extra info
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
