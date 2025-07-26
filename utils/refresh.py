# utils/refresh.py

import streamlit as st
import pandas as pd

def refresh_data():
    st.markdown("### 🔄 Refresh Dataset")

    if "raw_df" in st.session_state:
        if st.button("🔁 Refresh to Original Uploaded Dataset"):
            st.session_state.cleaned_df = st.session_state.raw_df.copy()
            st.success("Dataset has been refreshed to the original uploaded version.")
    else:
        st.warning("⚠️ No dataset has been uploaded yet.")
