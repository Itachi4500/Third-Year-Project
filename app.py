import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

import warnings
warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(page_title="🧠 Enhanced Data Analysis Assistant", layout="wide")
st.title("🧠 Enhanced Data Analysis Assistant")

# Upload CSV
file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Sidebar Preprocessing
    st.sidebar.header("⚙️ Preprocessing Options")
    dropna = st.sidebar.checkbox("Drop Missing Values", value=True)
    scale = st.sidebar.checkbox("Standard Scaling", value=True)
    encode = st.sidebar.checkbox("Label Encode Categorical", value=True)

    if dropna:
        df.dropna(inplace=True)
        st.sidebar.success("✅ Dropped missing rows")

    label_encoders = {}
    if encode:
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    if scale:
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

    st.subheader("🧹 Cleaned Data Sample")
    st.dataframe(df.head(), use_container_width=True)

    # Time Series Plotting
    st.subheader("📊 Time Series Visualization")
    datetime_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
    if datetime_cols:
        dt_col = st.selectbox("🕒 Datetime Column", datetime_cols)
        val_col = st.selectbox("📈 Value Column", df.columns.drop(dt_col))
        df[dt_col] = pd.to_datetime(df[dt_col])
        df.sort_values(dt_col, inplace=True)
        st.plotly_chart(px.line(df, x=dt_col, y=val_col, title=f"{val_col} over {dt_col}"))

    # Data Visualization
    st.subheader("📈 Exploratory Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        xcol = st.selectbox("X-axis", df.columns)
        ycol = st.selectbox("Y-axis", df.columns)
        st.plotly_chart(px.scatter(df, x=xcol, y=ycol, title=f"{ycol} vs {xcol}"))
    with col2:
        histcol = st.selectbox("Histogram Column", df.columns)
        st.plotly_chart(px.histogram(df, x=histcol, title=f"Distribution of {histcol}"))

    st.subheader("📌 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

    # Machine Learning
    st.subheader("🤖 ML Model Trainer")
    target = st.selectbox("🎯 Target Column", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    task = "classification" if y.nunique() <= 15 and y.dtype in [np.int64, np.int32] else "regression"
    st.info(f"Auto-Detected Task: **{task.capitalize()}**")

    model = None
    model_type = st.selectbox("🧠 Select Model", {
        "classification": ["RandomForestClassifier", "SVC", "LogisticRegression"],
        "regression": ["RandomForestRegressor", "SVR", "LinearRegression"]
    }[task])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "classification":
        model = {
            "RandomForestClassifier": RandomForestClassifier(),
            "SVC": SVC(),
            "LogisticRegression": LogisticRegression()
        }[model_type]
    else:
        model = {
            "RandomForestRegressor": RandomForestRegressor(),
            "SVR": SVR(),
            "LinearRegression": LinearRegression()
        }[model_type]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("📋 Model Evaluation")
    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        st.metric("✅ Accuracy", f"{acc*100:.2f}%")
        st.text("Classification Report")
        st.code(classification_report(y_test, y_pred))
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.metric("📉 RMSE", f"{rmse:.3f}")
        st.metric("📈 R² Score", f"{r2:.3f}")

    # Export
    st.subheader("📤 Export Cleaned Dataset")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv_data, "cleaned_data.csv", "text/csv")

    with st.expander("🧾 Power BI Integration Instructions"):
        st.markdown("""
        1. Save the CSV to OneDrive or SharePoint.
        2. Open your report in Power BI Service.
        3. Navigate to `Settings > Dataset > Schedule Refresh`.
        4. Enable and set refresh frequency as needed.
        """)
else:
    st.info("📂 Please upload a CSV file to get started.")
