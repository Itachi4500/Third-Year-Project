# utils/image.py

import plotly.graph_objects as go
import streamlit as st

def image_chart():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["A", "B", "C"], y=[10, 20, 30]))
    return fig

def export_image_chart(fig):
    st.markdown("### 📤 Export Chart")

    try:
        img_bytes = fig.to_image(format="png")  # Requires kaleido
        st.download_button("📸 Download as PNG", data=img_bytes, file_name="chart.png", mime="image/png")
    except Exception as e:
        st.warning(f"⚠️ PNG export failed: {e}. Only HTML export is available.")

    try:
        html = fig.to_html()
        st.download_button("📄 Download as HTML", data=html.encode(), file_name="chart.html", mime="text/html")
    except Exception as e:
        st.error(f"❌ HTML export failed: {e}")
