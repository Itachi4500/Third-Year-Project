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

# Sidebar Navigation
nav = st.sidebar.radio("📌 Navigation", [
    "Upload Data", "Data Cleaning", "Visualizations", "Model Training", "Export"
])

# Global session memory
if "df" not in st.session_state:
    st.session_state.df = None

# Upload Section
if nav == "Upload Data":
    file = st.file_uploader("📁 Upload CSV File", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("✅ File uploaded and loaded into memory!")
        st.dataframe(df.head())

# Cleaning Section
elif nav == "Data Cleaning":
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        st.subheader("🧹 Cleaning Options")
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

        st.session_state.df = df
        st.success("✅ Data cleaned and scaled.")
        st.dataframe(df.head())
    else:
        st.warning("📂 Please upload data first.")

# Visualization Section
elif nav == "Visualizations":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📈 Data Visualizations")

        datetime_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
        if datetime_cols:
            datetime_col = st.selectbox("Time Axis", datetime_cols)
            value_col = st.selectbox("Value Column", df.columns.drop(datetime_col))
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.sort_values(by=datetime_col, inplace=True)
            st.plotly_chart(px.line(df, x=datetime_col, y=value_col, title="Time Series"))

        col1, col2 = st.columns(2)

        with col1:
            xcol = st.selectbox("X-axis", df.columns)
            ycol = st.selectbox("Y-axis", df.columns)
            st.plotly_chart(px.scatter(df, x=xcol, y=ycol, title="Scatter Plot"))

        with col2:
            hist_col = st.selectbox("Histogram Column", df.columns)
            st.plotly_chart(px.histogram(df, x=hist_col, title="Histogram"))

        st.subheader("📊 Correlation Heatmap")
        st.plotly_chart(px.imshow(df.corr(), title="Correlation Matrix"))
    else:
        st.warning("📂 Please upload data first.")

# Model Training Section
elif nav == "Model Training":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("🤖 ML Model Builder")

        target = st.selectbox("🎯 Select target column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        task_type = "classification" if len(np.unique(y)) <= 15 and y.dtype in [np.int64, np.int32] else "regression"
        st.info(f"Auto-detected task: **{task_type.capitalize()}**")

        model_option = st.selectbox("Choose Model", {
            "classification": ["RandomForestClassifier", "SVM", "LogisticRegression"],
            "regression": ["RandomForestRegressor", "SVR", "LinearRegression"]
        }[task_type])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = None
        if task_type == "classification":
            model = {
                "RandomForestClassifier": RandomForestClassifier(),
                "SVM": SVC(),
                "LogisticRegression": LogisticRegression()
            }[model_option]
        else:
            model = {
                "RandomForestRegressor": RandomForestRegressor(),
                "SVR": SVR(),
                "LinearRegression": LinearRegression()
            }[model_option]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📋 Model Results")
        if task_type == "classification":
            st.write(f"✅ Accuracy: **{accuracy_score(y_test, y_pred)*100:.2f}%**")
            st.code(classification_report(y_test, y_pred), language="text")
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            st.write(f"📉 RMSE: **{rmse:.3f}**")
            st.write(f"📈 R² Score: **{r2:.3f}**")
    else:
        st.warning("📂 Please upload data first.")

# Export Section
elif nav == "Export":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📤 Export Cleaned Data")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "cleaned_data.csv", "text/csv")

        st.markdown("""
        ### 🔄 How to use in Power BI:
        1. Upload to OneDrive/SharePoint.
        2. In Power BI Service, go to `Dataset > Settings`.
        3. Set up a **Scheduled Refresh** to sync updates.
        """)
    else:
        st.warning("📂 Please upload data first.")
