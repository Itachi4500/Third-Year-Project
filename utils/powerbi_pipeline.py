import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

def powerbi_pipeline(df):
    st.title("📊 Power BI Style Dashboard")

    if df.empty:
        st.warning("⚠️ Uploaded dataset is empty. Please upload a valid file.")
        return

    overview_tab, transform_tab, dashboard_tab, insights_tab, ml_tab = st.tabs([
        "📁 Data Overview", "🛠️ Data Transformation", "📈 Custom Dashboard", "📌 Auto Insights", "🤖 ML Predictions"
    ])

    # --------------------- Tab 1: Data Overview ---------------------
    with overview_tab:
        show_data_overview(df)

    # --------------------- Tab 2: Data Transformation ---------------------
    with transform_tab:
        df = transform_data(df)

    # --------------------- Tab 3: Interactive Dashboard ---------------------
    with dashboard_tab:
        create_custom_dashboard(df)

    # --------------------- Tab 4: Auto Insights ---------------------
    with insights_tab:
        auto_insights(df)

    # --------------------- Tab 5: ML Predictions ---------------------
    with ml_tab:
        run_basic_ml(df)

    return df


# ------------------ Helper Functions ------------------

def show_data_overview(df):
    st.subheader("📁 Data Summary")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Data Types:", df.dtypes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("### 📊 Histograms")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)


def transform_data(df):
    st.subheader("🛠️ Normalize Numeric Columns")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    norm_cols = st.multiselect("Select columns to normalize (Min-Max)", num_cols)
    if st.button("Normalize Selected Columns"):
        for col in norm_cols:
            min_val, max_val = df[col].min(), df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
        st.success("✅ Normalization applied successfully!")

    st.subheader("📏 KPI Metrics")
    for col in num_cols:
        st.metric(f"Mean of {col}", round(df[col].mean(), 2))
        st.metric(f"Max of {col}", round(df[col].max(), 2))
        st.metric(f"Min of {col}", round(df[col].min(), 2))

    return df


def create_custom_dashboard(df):
    st.subheader("🎯 Create Custom Dashboard")
    filter_col = st.selectbox("🔍 Select column to filter", ["None"] + list(df.columns))

    if filter_col != "None":
        unique_vals = df[filter_col].dropna().unique()
        selected_vals = st.multiselect(f"Filter values for `{filter_col}`", unique_vals, default=list(unique_vals))
        df = df[df[filter_col].isin(selected_vals)]

    if df.empty:
        st.warning("No data to plot after filtering.")
        return

    x_axis = st.selectbox("Select X-axis Column", df.columns)
    y_axis = st.selectbox("Select Y-axis Column", df.select_dtypes(include=np.number).columns)
    chart_type = st.radio("Select Chart Type", ["Bar", "Line", "Scatter", "Area", "Pie"])

    fig = generate_chart(df, x_axis, y_axis, chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        export_chart(fig)
        save_load_config(x_axis, y_axis, chart_type, filter_col, selected_vals)


def generate_chart(df, x_axis, y_axis, chart_type):
    if chart_type == "Bar":
        return px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    elif chart_type == "Line":
        return px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
    elif chart_type == "Scatter":
        return px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    elif chart_type == "Area":
        return px.area(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
    elif chart_type == "Pie":
        return px.pie(df, names=x_axis, title=f"Pie Chart of {x_axis}")
    return None


def export_chart(fig):
    st.markdown("### 📤 Export Chart")
    img_bytes = fig.to_image(format="png")
    st.download_button("📸 Download as PNG", data=img_bytes, file_name="chart.png", mime="image/png")

    html = fig.to_html()
    st.download_button("📄 Download as HTML", data=html.encode(), file_name="chart.html", mime="text/html")


def save_load_config(x, y, chart_type, filter_col, selected_vals):
    config = {
        "x_axis": x,
        "y_axis": y,
        "chart_type": chart_type,
        "filter_column": filter_col,
        "selected_values": selected_vals if filter_col != "None" else None
    }

    config_json = json.dumps(config, indent=4)
    st.download_button("💾 Download Chart Config", config_json, file_name="chart_config.json", mime="application/json")

    uploaded_file = st.file_uploader("📤 Upload Config File", type="json")
    if uploaded_file:
        try:
            config_loaded = json.load(uploaded_file)
            st.success("✅ Config loaded:")
            st.json(config_loaded)
        except Exception as e:
            st.error(f"Failed to load config: {e}")


def auto_insights(df):
    st.subheader("📌 Auto Insights")
    num_cols = df.select_dtypes(include=np.number).columns
    if num_cols.empty:
        st.warning("No numeric columns available.")
        return
    col = st.selectbox("Select a numeric column", num_cols)
    if col:
        st.write(df[col].describe())


def run_basic_ml(df):
    st.subheader("🤖 ML Classifier: Random Forest")
    cols = df.select_dtypes(include=[np.number, 'object']).columns.tolist()

    label_col = st.selectbox("🎯 Target Column", cols)
    feature_cols = st.multiselect("🧩 Feature Columns", [c for c in cols if c != label_col])

    if st.button("🚀 Train Model"):
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
            st.dataframe(pd.DataFrame(report).transpose())

            # Feature importance
            st.markdown("### 🔍 Feature Importance")
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
            st.dataframe(importance_df.head(10))

        except Exception as e:
            st.error(f"🚫 Error training model: {e}")
