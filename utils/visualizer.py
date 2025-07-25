import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px

def show_visuals(df):
    st.subheader("📊 Visualization Dashboard")
    
    chart_type = st.selectbox("Select Chart Type", [
        "Histogram", "Heatmap", "Bar Chart", "Pie Chart", "Bubble Chart"
    ])
 # Histogram 
    if chart_type == "Histogram":
        column = st.selectbox("Select Numeric Column", df.select_dtypes(include=['int64', 'float64']).columns)
        bins = st.slider("Number of bins", 5, 100, 30)
        fig, ax = plt.subplots()
        sns.histplot(df[column], bins=bins, kde=True, ax=ax)
        st.pyplot(fig)
# Heatmap
    elif chart_type == "Heatmap":
        st.write("Correlation Matrix")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
# Bar Chart
    elif chart_type == "Bar Chart":
        col = st.selectbox("Select Categorical Column", df.select_dtypes(include='object').columns)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f"Bar Chart of {col}")
        st.pyplot(fig)
# Pie Chart
    elif chart_type == "Pie Chart":
        col = st.selectbox("Select Column for Pie Chart", df.select_dtypes(include='object').columns)
        pie_data = df[col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Pie Chart of {col}")
        st.pyplot(fig)

    elif chart_type == "Pie Chart":
        col = st.selectbox("Select Column for Pie Chart", df.select_dtypes(include='object').columns)

        # Calculate value counts
        pie_data = df[col].value_counts()
        total_categories = len(pie_data)

        # Limit to top N categories + others
        top_n = st.slider("Top N Categories", 3, min(20, total_categories), 10)
        top_values = pie_data[:top_n]
        if total_categories > top_n:
            top_values["Others"] = pie_data[top_n:].sum()

        labels = top_values.index
        sizes = top_values.values

        # Explode the largest section for effect
        explode = [0.05 if i == 0 else 0 for i in range(len(sizes))]

        # Colors
        colors = sns.color_palette("pastel", len(sizes))

        # Plot
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=140, explode=explode, colors=colors,
            textprops=dict(color="black", fontsize=10), pctdistance=0.85,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            shadow=True
        )

        # Draw circle for donut style
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        ax.set_title(f"Pie Chart of {col}", fontsize=14)
        ax.axis('equal')  # Equal aspect ratio ensures the pie is circular
        st.pyplot(fig)

# Line Chart

