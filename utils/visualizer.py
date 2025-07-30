import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def show_visuals(df):
    st.subheader("📊 Advanced Visualization Dashboard")

    chart_type = st.selectbox("Select Visualization Type", [
        "Histogram", "Heatmap", "Bar Chart", "Pie Chart", "Donut Chart",
        "Line Chart", "Scatter Plot", "Bubble Chart"
    ])

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if chart_type == "Histogram":
        col = st.selectbox("Choose Numeric Column", numeric_cols)
        bins = st.slider("Number of Bins", 5, 100, 30)
        fig = px.histogram(df, x=col, nbins=bins, marginal="box", title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap":
        st.write("Correlation Matrix (numeric only)")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        st.pyplot(fig)

    elif chart_type == "Bar Chart":
        col = st.selectbox("Choose Categorical Column", cat_cols)
        agg_col = st.selectbox("Optional: Aggregate by Numeric Column", ["None"] + numeric_cols)
        if agg_col == "None":
            fig = px.bar(df[col].value_counts().reset_index(), x="index", y=col,
                         labels={"index": col, col: "Count"}, title=f"Bar Chart of {col}")
        else:
            grouped = df.groupby(col)[agg_col].mean().reset_index()
            fig = px.bar(grouped, x=col, y=agg_col, title=f"Mean of {agg_col} by {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart" or chart_type == "Donut Chart":
        col = st.selectbox("Select Categorical Column for Pie", cat_cols)
        value_counts = df[col].value_counts()
        top_n = st.slider("Top Categories to Show", 3, min(20, len(value_counts)), 10)
        top_data = value_counts[:top_n]
        if len(value_counts) > top_n:
            top_data["Others"] = value_counts[top_n:].sum()
        fig = px.pie(
            names=top_data.index,
            values=top_data.values,
            hole=0.4 if chart_type == "Donut Chart" else 0.0,
            title=f"{chart_type} of {col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        x = st.selectbox("X-Axis", df.columns)
        y = st.selectbox("Y-Axis (numeric)", numeric_cols)
        color = st.selectbox("Optional Group By", ["None"] + cat_cols)
        fig = px.line(df, x=x, y=y, color=color if color != "None" else None, title=f"{y} over {x}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        x = st.selectbox("X-Axis", numeric_cols)
        y = st.selectbox("Y-Axis", numeric_cols)
        color = st.selectbox("Color By (Optional)", ["None"] + cat_cols)
        fig = px.scatter(df, x=x, y=y, color=df[color] if color != "None" else None,
                         title=f"Scatter Plot of {y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bubble Chart":
        x = st.selectbox("X-Axis", numeric_cols)
        y = st.selectbox("Y-Axis", numeric_cols)
        size = st.selectbox("Bubble Size", numeric_cols)
        color = st.selectbox("Group by Category", ["None"] + cat_cols)
        fig = px.scatter(df, x=x, y=y, size=size,
                         color=df[color] if color != "None" else None,
                         title=f"Bubble Chart: {y} vs {x} (size by {size})")
        st.plotly_chart(fig, use_container_width=True)
