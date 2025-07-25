import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def powerbi_pipeline(df):
    st.title("📊 Power BI Style Dashboard")

    # Tabs for full workflow
    overview_tab, transform_tab, dashboard_tab, insights_tab, ml_tab = st.tabs([
        "📁 Data Overview", "🛠️ Data Transformation", "📈 Custom Dashboard", "📌 Auto Insights", "🤖 ML Predictions"
    ])

    # --------------------- Tab 1: Data Overview ---------------------
    with overview_tab:
        st.subheader("📁 Data Summary")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write("Data Types:", df.dtypes)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("### 📊 Histograms")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in num_cols:
            fig = px.histogram(df, x=col, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)

    # --------------------- Tab 2: Transformation ---------------------
    with transform_tab:
        st.subheader("🛠️ Normalize Numeric Columns")
        norm_cols = st.multiselect("Select columns to normalize (Min-Max)", df.select_dtypes(include=np.number).columns)
        if st.button("Normalize Selected Columns"):
            for c in norm_cols:
                min_val, max_val = df[c].min(), df[c].max()
                df[c] = (df[c] - min_val) / (max_val - min_val)
            st.success("Normalization applied successfully!")

        st.subheader("📏 KPI Metrics")
        if len(num_cols) >= 1:
            for col in num_cols:
                st.metric(f"Mean of {col}", round(df[col].mean(), 2))
                st.metric(f"Max of {col}", round(df[col].max(), 2))
                st.metric(f"Min of {col}", round(df[col].min(), 2))

    # --------------------- Tab 3: Interactive Dashboard ---------------------
    with dashboard_tab:
        st.subheader("🎯 Create Custom Dashboard")

        st.markdown("#### 🔍 Add Filters (Optional)")
        filter_col = st.selectbox("Select column to filter", ["None"] + list(df.columns))
        if filter_col != "None":
            unique_vals = df[filter_col].dropna().unique()
            selected_vals = st.multiselect(f"Filter values for {filter_col}", unique_vals, default=unique_vals)
            df = df[df[filter_col].isin(selected_vals)]

        x_axis = st.selectbox("Select X-axis Column", df.columns)
        y_axis = st.selectbox("Select Y-axis Column", df.select_dtypes(include=np.number).columns)
        chart_type = st.radio("Select Chart Type", ["Bar", "Line", "Scatter", "Area", "Pie"])

        fig = None
        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Area":
            fig = px.area(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        elif chart_type == "Pie":
            fig = px.pie(df, names=x_axis, title=f"Pie Chart of {x_axis}")

        if fig:
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📤 Export Chart")
            if st.button("📸 Export as PNG"):
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="dashboard_chart.png",
                    mime="image/png"
                )

            if st.button("📄 Export as PDF (via HTML)"):
                html = fig.to_html()
                st.download_button(
                    label="Download PDF (HTML format)",
                    data=html.encode(),
                    file_name="chart.html",
                    mime="text/html"
                )

            # Save/Load Config
            config = {
                "x_axis": x_axis,
                "y_axis": y_axis,
                "chart_type": chart_type,
                "filter_col": filter_col,
                "selected_vals": selected_vals if filter_col != "None" else None
            }

            st.markdown("### 💾 Save / Load Chart Configuration")
            config_json = json.dumps(config, indent=4)
            st.download_button("💾 Download Config", config_json, file_name="dashboard_config.json", mime="application/json")

            uploaded_config = st.file_uploader("Upload Config File", type=["json"])
            if uploaded_config:
                config_loaded = json.load(uploaded_config)
                st.json(config_loaded)

    # --------------------- Tab 4: Auto Insights ---------------------
    with insights_tab:
        st.subheader("📌 Auto Insights")
        target = st.selectbox("Select a numeric column for descriptive stats", num_cols)
        if target:
            st.write(df[target].describe())

    # --------------------- Tab 5: Machine Learning ---------------------
    with ml_tab:
        st.subheader("🤖 Basic ML: Classification (Random Forest)")

        all_columns = df.select_dtypes(include=[np.number, 'object']).columns.tolist()
        label_col = st.selectbox("Select Target Column (classification)", all_columns)
        feature_cols = st.multiselect("Select Feature Columns", [col for col in all_columns if col != label_col])

        if st.button("Train Classifier"):
            try:
                X = pd.get_dummies(df[feature_cols])
                y = df[label_col]
                if y.dtype == 'O':
                    y = pd.factorize(y)[0]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            except Exception as e:
                st.error(f"Error in training model: {e}")

    return df
