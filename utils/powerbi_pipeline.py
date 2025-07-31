import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Insightify BI Dashboard")

# --- State Management ---
def initialize_state():
    """Initializes session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'transformed_df' not in st.session_state:
        st.session_state.transformed_df = pd.DataFrame()
    if 'dashboard_config' not in st.session_state:
        st.session_state.dashboard_config = {'kpis': [], 'charts': []}
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = pd.DataFrame()


# --- Main App Logic ---
def powerbi_pipeline():
    """Main function to run the Streamlit app."""
    initialize_state()

    st.sidebar.title("📈 Insightify BI")
    st.sidebar.write("Upload your data and start exploring!")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Reset state if a new file is uploaded
        if st.session_state.df.empty or not st.session_state.df.equals(df):
            st.session_state.df = df
            st.session_state.transformed_df = df.copy()
            st.session_state.filtered_df = df.copy()
            st.session_state.dashboard_config = {'kpis': [], 'charts': []}


    if st.session_state.df.empty:
        st.info("Please upload a CSV file to begin.")
        return

    # --- Sidebar Navigation and Global Filters ---
    page = st.sidebar.radio("Navigate", ["Data Overview", "Data Transformation", "Dashboard Builder", "Auto Insights", "ML Studio"])

    with st.sidebar.expander("🌐 Global Filters"):
        apply_global_filters()

    # --- Page Routing ---
    df_to_display = st.session_state.get('filtered_df', st.session_state.transformed_df)

    if page == "Data Overview":
        show_data_overview(st.session_state.df)
    elif page == "Data Transformation":
        show_data_transformation()
    elif page == "Dashboard Builder":
        create_custom_dashboard(df_to_display)
    elif page == "Auto Insights":
        auto_insights(df_to_display)
    elif page == "ML Studio":
        run_ml_studio(df_to_display)

# --- Helper Functions for Each Page ---

def show_data_overview(df):
    st.header("📁 Data Overview")
    st.write("Here's a first look at your raw dataset.")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.subheader("Column Information")
    info_df = pd.DataFrame({
        "DataType": df.dtypes,
        "NonNulls": df.count(),
        "Nulls": df.isnull().sum(),
        "UniqueValues": df.nunique()
    }).reset_index().rename(columns={'index': 'Column'})
    st.dataframe(info_df)

    st.subheader("Descriptive Statistics (Numeric Columns)")
    st.dataframe(df.describe().T)

def show_data_transformation():
    st.header("🛠️ Data Transformation (Power Query Style)")
    st.write("Modify your dataset. Changes made here will reflect across the app.")

    # Ensure we are working with the correct dataframe from session state
    df = st.session_state.transformed_df

    st.sidebar.subheader("Transformation Actions")
    action = st.sidebar.selectbox("Choose Action", ["Handle Missing Values", "Change Data Types", "Create Calculated Column", "Rename/Drop Columns"])

    if action == "Handle Missing Values":
        col = st.selectbox("Select Column", df.columns)
        method = st.selectbox("Method", ["Drop Rows with NA", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"])
        if st.button("Apply"):
            if method == "Drop Rows with NA":
                df.dropna(subset=[col], inplace=True)
            elif method == "Fill with Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Fill with Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "Fill with Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "Forward Fill":
                df[col].fillna(method='ffill', inplace=True)
            elif method == "Backward Fill":
                df[col].fillna(method='bfill', inplace=True)
            st.success(f"Applied '{method}' to column '{col}'.")
            st.rerun()

    elif action == "Change Data Types":
        col = st.selectbox("Select Column", df.columns)
        new_type = st.selectbox("New Type", ["object", "int64", "float64", "datetime64"])
        if st.button("Convert Type"):
            try:
                if new_type == 'datetime64':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(new_type)
                st.success(f"Converted '{col}' to {new_type}.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to convert: {e}")

    elif action == "Create Calculated Column":
        new_col_name = st.text_input("New Column Name")
        formula = st.text_input("Formula (e.g., df['col1'] * df['col2'])")
        if st.button("Create Column") and new_col_name and formula:
            try:
                df[new_col_name] = pd.eval(formula, engine='python', local_dict={'df': df})
                st.success(f"Created column '{new_col_name}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Invalid formula: {e}")

    elif action == "Rename/Drop Columns":
        action_type = st.radio("Select", ["Rename", "Drop"])
        if action_type == "Rename":
            col_to_rename = st.selectbox("Column to Rename", df.columns)
            new_name = st.text_input("New Name", value=col_to_rename)
            if st.button("Rename"):
                df.rename(columns={col_to_rename: new_name}, inplace=True)
                st.success(f"Renamed '{col_to_rename}' to '{new_name}'.")
                st.rerun()
        else: # Drop
            cols_to_drop = st.multiselect("Columns to Drop", df.columns)
            if st.button("Drop Selected"):
                df.drop(columns=cols_to_drop, inplace=True)
                st.success(f"Dropped columns: {cols_to_drop}.")
                st.rerun()

    st.session_state.transformed_df = df
    st.subheader("Transformed Data Preview")
    st.dataframe(st.session_state.transformed_df.head())

def apply_global_filters():
    """Applies filters to the transformed_df and stores it in filtered_df."""
    df = st.session_state.transformed_df.copy()

    filter_cols = st.multiselect("Select columns to filter by", df.columns, key="global_filter_cols")

    filtered_df = df
    for col in filter_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            selected_range = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val), key=f"filter_{col}")
            filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]
        else:
            unique_vals = df[col].dropna().unique()
            selected_vals = st.multiselect(f"Filter {col}", unique_vals, default=list(unique_vals), key=f"filter_{col}")
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.session_state.filtered_df = filtered_df

def create_custom_dashboard(df):
    st.header("📈 Dashboard Builder")
    st.write("Build your custom dashboard by adding KPIs and charts.")

    with st.expander("Add Components to Dashboard"):
        component_type = st.selectbox("Component Type", ["KPI", "Chart"])
        if component_type == "KPI":
            col = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
            agg = st.selectbox("Aggregation", ["Sum", "Average", "Count", "Max", "Min"])
            if st.button("Add KPI"):
                st.session_state.dashboard_config['kpis'].append({'col': col, 'agg': agg})
                st.rerun()
        else: # Chart
            chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie"])
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis_cols = df.select_dtypes(include=np.number).columns
            y_axis = st.selectbox("Y-axis (numeric)", y_axis_cols if chart_type != "Pie" else [None], disabled=chart_type=="Pie")
            if st.button("Add Chart"):
                st.session_state.dashboard_config['charts'].append({'type': chart_type, 'x': x_axis, 'y': y_axis})
                st.rerun()

    # Render KPIs
    if st.session_state.dashboard_config['kpis']:
        st.subheader("Key Performance Indicators")
        num_kpis = len(st.session_state.dashboard_config['kpis'])
        kpi_cols = st.columns(num_kpis)
        for i, kpi in enumerate(st.session_state.dashboard_config['kpis']):
            with kpi_cols[i]:
                if kpi['agg'] == 'Sum': value = df[kpi['col']].sum()
                elif kpi['agg'] == 'Average': value = df[kpi['col']].mean()
                elif kpi['agg'] == 'Count': value = df[kpi['col']].count()
                elif kpi['agg'] == 'Max': value = df[kpi['col']].max()
                elif kpi['agg'] == 'Min': value = df[kpi['col']].min()
                st.metric(f"{kpi['agg']} of {kpi['col']}", f"{value:,.2f}")

    # Render Charts
    if st.session_state.dashboard_config['charts']:
        st.subheader("Charts")
        num_charts = len(st.session_state.dashboard_config['charts'])
        chart_cols = st.columns(2)
        for i, chart in enumerate(st.session_state.dashboard_config['charts']):
            with chart_cols[i % 2]:
                try:
                    if chart['type'] == "Bar": fig = px.bar(df, x=chart['x'], y=chart['y'], title=f"{chart['y']} by {chart['x']}")
                    elif chart['type'] == "Line": fig = px.line(df.sort_values(chart['x']), x=chart['x'], y=chart['y'], title=f"{chart['y']} over {chart['x']}")
                    elif chart['type'] == "Scatter": fig = px.scatter(df, x=chart['x'], y=chart['y'], title=f"{chart['y']} vs {chart['x']}")
                    elif chart['type'] == "Pie": fig = px.pie(df, names=chart['x'], title=f"Distribution of {chart['x']}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not plot {chart['type']} chart: {e}")

    if st.button("Clear Dashboard"):
        st.session_state.dashboard_config = {'kpis': [], 'charts': []}
        st.rerun()

def auto_insights(df):
    st.header("📌 Auto Insights")
    st.write("Let AI find interesting patterns in your data.")

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

        # Top correlations
        corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
        corr_unstacked = corr_unstacked[corr_unstacked != 1.0]
        st.write("**Top 5 Positive Correlations:**")
        st.dataframe(corr_unstacked.head(5))
        st.write("**Top 5 Negative Correlations:**")
        st.dataframe(corr_unstacked.tail(5))
    else:
        st.info("Not enough numeric columns for correlation analysis.")

    # Outlier Detection
    st.subheader("Outlier Detection")
    col_to_check = st.selectbox("Select a numeric column for outlier analysis", numeric_cols)
    if col_to_check:
        Q1 = df[col_to_check].quantile(0.25)
        Q3 = df[col_to_check].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col_to_check] < (Q1 - 1.5 * IQR)) | (df[col_to_check] > (Q3 + 1.5 * IQR))]
        st.write(f"Found **{len(outliers)}** potential outliers in '{col_to_check}' based on the IQR method.")
        if not outliers.empty:
            st.dataframe(outliers)

def run_ml_studio(df):
    st.header("🤖 ML Studio")
    st.write("Train a model and make predictions.")

    problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"])

    target_col = st.selectbox("🎯 Select Target Column", df.columns)
    feature_cols = st.multiselect("🧩 Select Feature Columns", [c for c in df.columns if c != target_col])

    if not target_col or not feature_cols:
        st.warning("Please select target and feature columns.")
        return

    # Preprocessing
    X = df[feature_cols]
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True) # One-hot encode categorical features

    if problem_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
        y, class_names = pd.factorize(y) # Label encode target
        st.session_state.ml_class_names = class_names
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            if problem_type == "Classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred))

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

            else: # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.subheader("Regression Metrics")
                col1, col2 = st.columns(2)
                col1.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.3f}")
                col2.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.3f}")

                st.subheader("Actual vs. Predicted Values")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal', line=dict(dash='dash')))
                fig.update_layout(xaxis_title="Actual Values", yaxis_title="Predicted Values")
                st.plotly_chart(fig, use_container_width=True)

            # Feature Importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            st.dataframe(importance_df)

            st.session_state.ml_model = model
            st.session_state.ml_features = X.columns
            st.session_state.ml_problem_type = problem_type

    # What-if analysis
    if 'ml_model' in st.session_state:
        st.subheader("🔮 What-If Analysis / Live Prediction")
        with st.form("prediction_form"):
            inputs = {}
            for col in st.session_state.ml_features:
                # Align input types with original data if possible
                original_col_name = col.split('_')[0]
                if original_col_name in df.columns and pd.api.types.is_numeric_dtype(df[original_col_name]):
                     inputs[col] = st.number_input(f"Input for {col}", value=float(df[original_col_name].mean()))
                else:
                     inputs[col] = st.number_input(f"Input for {col}", value=0)


            submitted = st.form_submit_button("Predict")
            if submitted:
                input_df = pd.DataFrame([inputs])
                prediction = st.session_state.ml_model.predict(input_df)[0]
                
                if st.session_state.ml_problem_type == "Classification" and 'ml_class_names' in st.session_state:
                    prediction_label = st.session_state.ml_class_names[prediction]
                    st.success(f"**Predicted Outcome: `{prediction_label}`**")
                else:
                    st.success(f"**Predicted Outcome: `{prediction:,.2f}`**")


if __name__ == "__main__":
    # The main function should be called without arguments.
    # It uses st.session_state to manage data internally.
    powerbi_pipeline()
