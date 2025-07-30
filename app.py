# enhanced_data_analysis_assistant.py

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
st.set_page_config(page_title="Data Analysis Assistant", layout="wide")
st.title("🧠 Enhanced Data Analysis Assistant")

file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    st.subheader("🧹 Cleaning")
    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
        st.success("Missing rows dropped.")

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    # Time series detection
    st.subheader("📊 Time Series Visualization")
    datetime_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
    if datetime_cols:
        datetime_col = st.selectbox("Select datetime column", datetime_cols)
        value_col = st.selectbox("Select value column", df.columns.drop(datetime_col))
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.sort_values(by=datetime_col, inplace=True)
        st.plotly_chart(px.line(df, x=datetime_col, y=value_col, title="Time Series Plot"))

    # Interactive Visualizations
    st.subheader("📈 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        xcol = st.selectbox("X-axis", df.columns)
        ycol = st.selectbox("Y-axis", df.columns)
        st.plotly_chart(px.scatter(df, x=xcol, y=ycol, title="Scatter Plot"))

    with col2:
        hist_col = st.selectbox("Histogram column", df.columns)
        st.plotly_chart(px.histogram(df, x=hist_col, title="Histogram"))

    st.plotly_chart(px.imshow(df.corr(), title="📊 Correlation Heatmap"))

    # ML Modeling
    st.subheader("🤖 ML Model Training")
    target = st.selectbox("🎯 Select target column", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    task_type = "classification" if len(y.unique()) <= 15 and y.dtype in [np.int64, np.int32] else "regression"
    st.info(f"Auto-detected Task: **{task_type.capitalize()}**")

    model_option = st.selectbox("Select Model", {
        "classification": ["RandomForestClassifier", "SVM", "LogisticRegression"],
        "regression": ["RandomForestRegressor", "SVR", "LinearRegression"]
    }[task_type])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None
    if task_type == "classification":
        if model_option == "RandomForestClassifier":
            model = RandomForestClassifier()
        elif model_option == "SVM":
            model = SVC()
        elif model_option == "LogisticRegression":
            model = LogisticRegression()
    else:
        if model_option == "RandomForestRegressor":
            model = RandomForestRegressor()
        elif model_option == "SVR":
            model = SVR()
        elif model_option == "LinearRegression":
            model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📋 Model Evaluation")
    if task_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        st.write(f"✅ Accuracy: **{acc*100:.2f}%**")
        st.code(classification_report(y_test, y_pred), language="text")
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write(f"📉 RMSE: **{rmse:.3f}**")
        st.write(f"📈 R² Score: **{r2:.3f}**")

    # Export
    st.subheader("📤 Export for Power BI")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")

    st.markdown("""
    ### 🔄 Schedule Power BI Refresh (Pro)
    1. Save this CSV to OneDrive or SharePoint.
    2. In [Power BI Service](https://app.powerbi.com), open your report.
    3. Go to `Settings > Dataset > Schedule Refresh`.
    4. Enable refresh and set the frequency.
    """)

else:
    st.warning("📂 Please upload a CSV file to continue.")
