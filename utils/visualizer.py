import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def show_visuals(df):
    st.title("📊 Advanced Visualization Dashboard")

    # Column Detection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    with st.sidebar:
        st.header("🧭 Visualization Navigator")
        chart_type = st.selectbox("Choose Chart Type", [
            "Histogram", "Heatmap", "Bar Chart", "Pie Chart", "Donut Chart",
            "Line Chart", "Scatter Plot", "Bubble Chart"
        ])
        st.markdown("---")

    # Show different charts based on selection
    st.subheader(f"🗂️ {chart_type}")

    # if chart_type == "Histogram":
    #     if numeric_cols:
    #         col = st.sidebar.selectbox("Numeric Column", numeric_cols)
    #         bins = st.sidebar.slider("Number of Bins", 5, 100, 30)
    #         fig = px.histogram(df, x=col, nbins=bins, marginal="box", title=f"Histogram of {col}")
    #         st.plotly_chart(fig, use_container_width=True)
        if chart_type == "Histogram":
            column = st.selectbox("Select Numeric Column", df.select_dtypes(include=['int64', 'float64']).columns)
            bins = st.slider("Number of bins", 5, 100, 30)
            fig, ax = plt.subplots()
            sns.histplot(df[column], bins=bins, kde=True, ax=ax)
            st.pyplot(fig)
        
        else:
            st.warning("No numeric columns available for histogram.")

    elif chart_type == "Heatmap":
        if len(numeric_cols) > 1:
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            st.pyplot(fig)
        else:
            st.warning("At least two numeric columns required for heatmap.")

    elif chart_type == "Bar Chart":
        if cat_cols:
            col = st.sidebar.selectbox("Categorical Column", cat_cols)
            agg_col = st.sidebar.selectbox("Aggregate by (optional)", ["None"] + numeric_cols)
            if agg_col == "None":
                count_data = df[col].value_counts().reset_index()
                fig = px.bar(count_data, x="index", y=col,
                             labels={"index": col, col: "Count"},
                             title=f"Bar Chart of {col}")
            else:
                grouped = df.groupby(col)[agg_col].mean().reset_index()
                fig = px.bar(grouped, x=col, y=agg_col, title=f"Mean {agg_col} by {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns available for bar chart.")

    elif chart_type in ["Pie Chart", "Donut Chart"]:
        if cat_cols:
            col = st.sidebar.selectbox("Categorical Column", cat_cols)
            value_counts = df[col].value_counts()
            top_n = st.sidebar.slider("Top Categories", 3, min(20, len(value_counts)), 10)
            top_data = value_counts[:top_n]
            if len(value_counts) > top_n:
                top_data["Others"] = value_counts[top_n:].sum()
            fig = px.pie(
                names=top_data.index,
                values=top_data.values,
                hole=0.4 if chart_type == "Donut Chart" else 0.0,
                title=f"{chart_type} of {col}",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detail view
            selected = st.selectbox("🔍 Show details for category", top_data.index)
            filtered = df[df[col] == selected] if selected != "Others" else df[df[col].isin(value_counts[top_n:].index)]
            st.markdown(f"### 📋 Data for '{selected}'")
            st.write(f"**Total Records:** {len(filtered)}")
            st.dataframe(filtered.head(10))
        else:
            st.warning("No categorical columns available for pie/donut chart.")

    elif chart_type == "Line Chart":
        if numeric_cols:
            x = st.sidebar.selectbox("X-Axis", df.columns)
            y = st.sidebar.selectbox("Y-Axis (numeric)", numeric_cols)
            color = st.sidebar.selectbox("Group by (optional)", ["None"] + cat_cols)
            fig = px.line(df, x=x, y=y, color=color if color != "None" else None, title=f"{y} over {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Numeric columns required for line chart.")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x = st.sidebar.selectbox("X-Axis", numeric_cols)
            y = st.sidebar.selectbox("Y-Axis", [col for col in numeric_cols if col != x])
            color = st.sidebar.selectbox("Color By", ["None"] + cat_cols)
            fig = px.scatter(df, x=x, y=y, color=df[color] if color != "None" else None,
                             title=f"Scatter Plot of {y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least two numeric columns required for scatter plot.")

    elif chart_type == "Bubble Chart":
        if len(numeric_cols) >= 3:
            x = st.sidebar.selectbox("X-Axis", numeric_cols)
            y = st.sidebar.selectbox("Y-Axis", [col for col in numeric_cols if col != x])
            size = st.sidebar.selectbox("Bubble Size", [col for col in numeric_cols if col not in [x, y]])
            color = st.sidebar.selectbox("Group by", ["None"] + cat_cols)
            fig = px.scatter(df, x=x, y=y, size=size,
                             color=df[color] if color != "None" else None,
                             title=f"Bubble Chart of {y} vs {x} (size by {size})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least three numeric columns required for bubble chart.")
