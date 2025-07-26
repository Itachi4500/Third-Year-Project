import streamlit as st
import pandas as pd

def upload_data(uploaded_file):
  if "uploaded_df" in st.session_state:
    df = st.session_state.uploaded_df
    # Proceed with cleaning or EDA
else:
    st.warning("⚠️ Please upload a dataset first.")

    try:
        file_name = uploaded_file.name
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.warning("⚠️ Please upload a valid CSV or Excel file.")
            return None

        st.session_state.uploaded_df = df
        st.success("✅ File uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None

