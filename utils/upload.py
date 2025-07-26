# utils/upload.py

import streamlit as st
import pandas as pd

def upload_data():
    uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name

        try:
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.warning("⚠️ Please upload a valid CSV or Excel file.")
                return None

            # Save to session state
            st.session_state.uploaded_df = df
            st.success("✅ File uploaded and saved successfully!")
            return df

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return None

    return None
