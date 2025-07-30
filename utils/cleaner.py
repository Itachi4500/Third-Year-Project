import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore

def clean_data(df):
    st.subheader("🧹 Data Cleaning Dashboard")

    # Initial check
    if df.empty:
        st.error("❌ The uploaded dataset is empty.")
        return df

    # 1. Handle Missing Values
    st.markdown("### 🔍 Handle Missing Values")
    missing_action = st.radio("Choose missing value handling method", ["None", "Fill NA", "Drop NA"])

    if missing_action == "Fill NA":
        method = st.selectbox("Fill Method", ["Forward Fill", "Backward Fill", "Mean", "Median", "Zero"])
        if method == "Forward Fill":
            df = df.fillna(method='ffill')
        elif method == "Backward Fill":
            df = df.fillna(method='bfill')
        elif method == "Mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif method == "Median":
            df = df.fillna(df.median(numeric_only=True))
        elif method == "Zero":
            df = df.fillna(0)
        st.success(f"Missing values filled using: {method}")

    elif missing_action == "Drop NA":
        original_size = df.shape[0]
        df = df.dropna()
        dropped = original_size - df.shape[0]
        st.success(f"Dropped {dropped} rows with missing values.")

    st.write("### Remaining Missing Values:")
    st.dataframe(df.isnull().sum())

    # 2. Outlier Detection and Removal
    st.markdown("### ⚠️ Outlier Detection & Removal")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_cols:
        outlier_col = st.selectbox("Select Numeric Column", numeric_cols, key="outlier_col")
        method = st.selectbox("Outlier Detection Method", ["IQR Method", "Z-Score Method"], key="outlier_method")

        if st.button("Remove Outliers"):
            original_size = df.shape[0]
            if method == "IQR Method":
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[outlier_col] >= Q1 - 1.5 * IQR) & (df[outlier_col] <= Q3 + 1.5 * IQR)]
            else:
                z_scores = zscore(df[outlier_col])
                df = df[(np.abs(z_scores) < 3)]
            new_size = df.shape[0]
            st.success(f"{original_size - new_size} outliers removed using {method}. Rows reduced from {original_size} to {new_size}.")
    else:
        st.warning("No numeric columns available for outlier detection.")

    if df.empty:
        st.error("⚠️ DataFrame is empty after cleaning. Please upload a valid dataset.")
        return df

    # 3. Column Filtering
    st.markdown("### 🧠 Column Selection for Analysis")
    selected_cols = st.multiselect("Select columns to keep", df.columns.tolist(), default=df.columns.tolist())
    df = df[selected_cols]
    st.write("Filtered DataFrame Preview:")
    st.dataframe(df.head())

    # 4. Duplicate Cleaned Dataset to Session State
    st.markdown("### ✅ Save Cleaned Dataset")
    cleaned_df = df.copy()
    st.session_state.cleaned_df = cleaned_df
    st.success("Cleaned dataset saved for further analysis.")

    # 5. Export Option
    st.markdown("### 📤 Export Cleaned Data")
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")

    return cleaned_df
