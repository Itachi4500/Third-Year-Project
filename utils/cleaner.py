import streamlit as st
import pandas as pd
import numpy as np

def clean_data(df):
    st.subheader("🧹 Data Cleaning Dashboard")

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
        st.success("Missing values filled using: " + method)

    elif missing_action == "Drop NA":
        df = df.dropna()
        st.success("Rows with missing values dropped.")

    st.write("### Remaining Missing Values:")
    st.dataframe(df.isnull().sum())

    # 2. Outlier Detection and Removal
    st.markdown("### ⚠️ Outlier Detection & Removal")
    outlier_col = st.selectbox("Select Numeric Column", df.select_dtypes(include=['float64', 'int64']).columns, key="outlier_col")
    method = st.selectbox("Outlier Detection Method", ["IQR Method", "Z-Score Method"], key="outlier_method")

    if st.button("Remove Outliers"):
        original_size = df.shape[0]
        if method == "IQR Method":
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[outlier_col] >= Q1 - 1.5 * IQR) & (df[outlier_col] <= Q3 + 1.5 * IQR)]
        else:  # Z-Score Method
            from scipy.stats import zscore
            z_scores = zscore(df[outlier_col])
            df = df[(np.abs(z_scores) < 3)]
        new_size = df.shape[0]
        st.success(f"Outliers removed using {method}. Rows reduced from {original_size} to {new_size}.")

    # 3. Column Filtering (Interactive)
    st.markdown("### 🧠 Column Selection for Analysis")
    selected_cols = st.multiselect("Select columns to keep", df.columns.tolist(), default=df.columns.tolist())
    df = df[selected_cols]
    st.write("Filtered DataFrame Preview:")
    st.dataframe(df.head())

    # 4. Duplicate the cleaned dataset
    st.markdown("### ✅ Save Cleaned Dataset")
    cleaned_df = df.copy()
    st.session_state.cleaned_df = cleaned_df
    st.success("Cleaned dataset duplicated for further analysis and modeling.")

    return cleaned_df

