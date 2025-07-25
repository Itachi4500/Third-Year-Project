import streamlit as st
import pandas as pd

def export_data(df):
    st.download_button("Download as CSV", df.to_csv(index=False), "cleaned_data.csv", "text/csv")
