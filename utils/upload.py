# utils/upload.py

import streamlit as st
import pandas as pd

def upload_data():
    uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.uploaded_df = df
            st.success("✅ Dataset uploaded successfully.")
            return df

        except Exception as e:
            st.error(f"❌ Error uploading file: {e}")
            return None

    return None

