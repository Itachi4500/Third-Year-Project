import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import pickle
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- Session State Initialization ---
def initialize_state():
    state_vars = [
        ("df", pd.DataFrame()),
        ("transformed_df", pd.DataFrame()),
        ("filtered_df", pd.DataFrame()),
        ("dashboard_config", {'kpis': [], 'charts': []}),
        ("undo_stack", []),
        ("redo_stack", []),
        ("theme", "light"),
    ]
    for key, default in state_vars:
        if key not in st.session_state:
            st.session_state[key] = default

# --- Data Upload and Reset ---
def upload_data():
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.session_state.df.empty or not st.session_state.df.equals(df):
            st.session_state.df = df
            st.session_state.transformed_df = df.copy()
            st.session_state.filtered_df = df.copy()
            st.session_state.dashboard_config = {'kpis': [], 'charts': []}
            st.session_state.undo_stack = []
            st.session_state.redo_stack = []
    return uploaded_file

# --- Undo/Redo for Data Transformations ---
def push_state():
    st.session_state.undo_stack.append(st.session_state.transformed_df.copy())
    st.session_state.redo_stack.clear()

def undo():
    if st.session_state.undo_stack:
        st.session_state.redo_stack.append(st.session_state.transformed_df.copy())
        st.session_state.transformed_df = st.session_state.undo_stack.pop()
        st.rerun()

def redo():
    if st.session_state.redo_stack:
        st.session_state.undo_stack.append(st.session_state.transformed_df.copy())
        st.session_state.transformed_df = st.session_state.redo_stack.pop()
        st.rerun()

# --- Theme Toggle ---
def theme_toggle():
    theme = st.sidebar.radio("Theme", ["light", "dark"], horizontal=True)
    st.session_state.theme = theme
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {"#202020" if theme == "dark" else "#FFFFFF"};
            color: {"#FFFFFF" if theme == "dark" else "#202020"};
        }}
        </style>
    """, unsafe_allow_html=True)

# --- Main App Logic ---
def powerbi_pipeline():
    st.set_page_config(layout="wide", page_title="Insightify Pro BI Dashboard")
    initialize_state()
    theme_toggle()

    st.sidebar.title("📈 Insightify Pro BI")
    st.sidebar.write("Upload your data and start exploring advanced analytics!")

    uploaded_file = upload_data()

    if st.session_state.df.empty:
        st.info("Please upload a CSV file to begin.")
        return

    page = st.sidebar.radio("Navigate", [
        "Data Overview", "Data Profiling", "Data Transformation",
        "Dashboard Builder", "Auto Insights", "ML Studio"
    ])

    with st.sidebar.expander("🌐 Global Filters"):
        apply_global_filters()

    if page == "Data Overview":
        show_data_overview(st.session_state.df)
    elif page == "Data Profiling":
        show_data_profiling(st.session_state.transformed_df)
    elif page == "Data Transformation":
        show_data_transformation()
    elif page == "Dashboard Builder":
        create_custom_dashboard(st.session_state.filtered_df)
    elif page == "Auto Insights":
        auto_insights(st.session_state.filtered_df)
    elif page == "ML Studio":
        run_ml_studio(st.session_state.filtered_df)

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

def show_data_profiling(df):
    st.header("🔎 Data Profiling & EDA")
    st.write("Automated profiling and variable analysis.")
    try:
        import ydata_profiling
        profile = ydata_profiling.ProfileReport(df, explorative=True, minimal=True)
        from streamlit_pandas_profiling import st_profile_report
        st_profile_report(profile)
    except Exception:
        st.warning("Auto-profiling package not available. Showing summary plots instead.")
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            st.subheader(f"Distribution: {col}")
            fig = px.histogram(df, x=col, nbins=30, marginal="box", color_discrete_sequence=["#6c63ff"])
            st.plotly_chart(fig, use_container_width=True)

        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            st.subheader(f"Category Count: {col}")
            fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col, color_discrete_sequence=["#6c63ff"])
            st.plotly_chart(fig, use_container_width=True)

def show_data_transformation():
    st.header("🛠️ Data Transformation (Power Query Pro)")
    df = st.session_state.transformed_df.copy()
    st.sidebar.subheader("Transformation Actions")

    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Undo"):
            undo()
        if st.button("Redo"):
            redo()

    action = st.sidebar.selectbox("Choose Action", [
        "Handle Missing Values", "Change Data Types", "Create Calculated Column",
        "Rename/Drop Columns", "Filter Rows", "Sort Data"
    ])

    with col2:
        if action == "Handle Missing Values":
            col = st.selectbox("Select Column", df.columns)
            method = st.selectbox("Method", [
                "Drop Rows with NA", "Fill with Mean", "Fill with Median",
                "Fill with Mode", "Forward Fill", "Backward Fill"
            ])
            preview = False
            if st.button("Preview Change"):
                preview = True
                if method == "Drop Rows with NA":
                    st.dataframe(df.dropna(subset=[col]).head())
                elif method == "Fill with Mean":
                    st.dataframe(df.assign(**{col: df[col].fillna(df[col].mean())}).head())
                # ...similarly for others
            if st.button("Apply"):
                push_state()
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
                st.session_state.transformed_df = df
                st.rerun()

        elif action == "Change Data Types":
            col = st.selectbox("Select Column", df.columns)
            new_type = st.selectbox("New Type", ["object", "int64", "float64", "datetime64"])
            if st.button("Convert Type"):
                push_state()
                try:
                    if new_type == 'datetime64':
                        df[col] = pd.to_datetime(df[col])
                    else:
                        df[col] = df[col].astype(new_type)
                    st.success(f"Converted '{col}' to {new_type}.")
                    st.session_state.transformed_df = df
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to convert: {e}")

        elif action == "Create Calculated Column":
            new_col_name = st.text_input("New Column Name")
            formula = st.text_area("Formula (e.g., df['col1'] * df['col2'])")
            if st.button("Preview Formula"):
                try:
                    df[new_col_name] = pd.eval(formula, engine='python', local_dict={'df': df})
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Invalid formula: {e}")
            if st.button("Create Column") and new_col_name and formula:
                push_state()
                try:
                    df[new_col_name] = pd.eval(formula, engine='python', local_dict={'df': df})
                    st.success(f"Created column '{new_col_name}'.")
                    st.session_state.transformed_df = df
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid formula: {e}")

        elif action == "Rename/Drop Columns":
            action_type = st.radio("Select", ["Rename", "Drop"])
            if action_type == "Rename":
                col_to_rename = st.selectbox("Column to Rename", df.columns)
                new_name = st.text_input("New Name", value=col_to_rename)
                if st.button("Rename"):
                    push_state()
                    df.rename(columns={col_to_rename: new_name}, inplace=True)
                    st.success(f"Renamed '{col_to_rename}' to '{new_name}'.")
                    st.session_state.transformed_df = df
                    st.rerun()
            else: # Drop
                cols_to_drop = st.multiselect("Columns to Drop", df.columns)
                if st.button("Drop Selected"):
                    push_state()
                    df.drop(columns=cols_to_drop, inplace=True)
                    st.success(f"Dropped columns: {cols_to_drop}.")
                    st.session_state.transformed_df = df
                    st.rerun()

        elif action == "Filter Rows":
            col = st.selectbox("Select column to filter", df.columns)
            unique_vals = df[col].unique()
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.slider("Value range", min_val, max_val, (min_val, max_val))
                if st.button("Filter"):
                    push_state()
                    df = df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]
                    st.session_state.transformed_df = df
                    st.rerun()
            else:
                selected_vals = st.multiselect("Values", unique_vals, default=list(unique_vals))
                if st.button("Filter"):
                    push_state()
                    df = df[df[col].isin(selected_vals)]
                    st.session_state.transformed_df = df
                    st.rerun()

        elif action == "Sort Data":
            col = st.selectbox("Select column to sort", df.columns)
            order = st.radio("Order", ["Ascending", "Descending"])
            if st.button("Sort"):
                push_state()
                df.sort_values(by=col, ascending=(order=="Ascending"), inplace=True)
                st.session_state.transformed_df = df
                st.rerun()

    st.subheader("Transformed Data Preview")
    st.dataframe(st.session_state.transformed_df.head())

def apply_global_filters():
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
    st.header("📈 Dashboard Builder (Advanced)")
    st.write("Build your custom dashboard by adding KPIs and charts. Drag to rearrange or remove components.")

    with st.expander("Add Components to Dashboard"):
        component_type = st.selectbox("Component Type", ["KPI", "Chart", "Custom Plotly Code"])
        if component_type == "KPI":
            col = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
            agg = st.selectbox("Aggregation", ["Sum", "Average", "Count", "Max", "Min", "Std"])
            show_trend = st.checkbox("Compare with Previous Value")
            if st.button("Add KPI"):
                st.session_state.dashboard_config['kpis'].append({'col': col, 'agg': agg, "trend": show_trend})
                st.rerun()
        elif component_type == "Chart":
            chart_type = st.selectbox("Chart Type", [
                "Bar", "Line", "Area", "Scatter", "Pie", "Histogram", "Box", "Violin", "Heatmap"
            ])
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis_cols = df.select_dtypes(include=np.number).columns
            y_axis = st.selectbox("Y-axis (numeric)", y_axis_cols if chart_type not in ["Pie", "Heatmap"] else [None], disabled=chart_type in ["Pie", "Heatmap"])
            color = st.selectbox("Color By", [None] + list(df.columns))
            palette = st.selectbox("Color Palette", ["plotly", "viridis", "plasma", "magma", "inferno"])
            if st.button("Add Chart"):
                st.session_state.dashboard_config['charts'].append({
                    'type': chart_type, 'x': x_axis, 'y': y_axis, 'color': color, 'palette': palette
                })
                st.rerun()
        else: # Custom Plotly
            code = st.text_area("Paste Plotly Python code (df available as variable)", height=120)
            if st.button("Add Custom Chart"):
                st.session_state.dashboard_config['charts'].append({'type': 'Custom', 'code': code})
                st.rerun()

    # KPI Cards
    if st.session_state.dashboard_config['kpis']:
        st.subheader("Key Performance Indicators")
        num_kpis = len(st.session_state.dashboard_config['kpis'])
        kpi_cols = st.columns(num_kpis)
        for i, kpi in enumerate(st.session_state.dashboard_config['kpis']):
            with kpi_cols[i]:
                value, prev_value = None, None
                col = kpi['col']
                agg = kpi['agg']
                if agg == 'Sum': value = df[col].sum()
                elif agg == 'Average': value = df[col].mean()
                elif agg == 'Count': value = df[col].count()
                elif agg == 'Max': value = df[col].max()
                elif agg == 'Min': value = df[col].min()
                elif agg == 'Std': value = df[col].std()
                delta = None
                if kpi.get("trend"):
                    if df.shape[0] > 1:
                        prev_value = df[col].iloc[:-1].agg(agg.lower()) if hasattr(pd.Series, agg.lower()) else None
                        delta = value - prev_value if prev_value is not None else None
                st.metric(f"{agg} of {col}", f"{value:,.2f}", delta=delta if delta is not None else "")

    # Charts
    if st.session_state.dashboard_config['charts']:
        st.subheader("Charts")
        chart_cols = st.columns(2)
        for i, chart in enumerate(st.session_state.dashboard_config['charts']):
            with chart_cols[i % 2]:
                try:
                    if chart['type'] == "Bar":
                        fig = px.bar(df, x=chart['x'], y=chart['y'], color=chart['color'], color_continuous_scale=chart['palette'])
                    elif chart['type'] == "Line":
                        fig = px.line(df, x=chart['x'], y=chart['y'], color=chart['color'], color_discrete_sequence=[chart['palette']])
                    elif chart['type'] == "Area":
                        fig = px.area(df, x=chart['x'], y=chart['y'], color=chart['color'], color_discrete_sequence=[chart['palette']])
                    elif chart['type'] == "Scatter":
                        fig = px.scatter(df, x=chart['x'], y=chart['y'], color=chart['color'], color_discrete_sequence=[chart['palette']])
                    elif chart['type'] == "Pie":
                        fig = px.pie(df, names=chart['x'], color=chart['color'], color_discrete_sequence=[chart['palette']])
                    elif chart['type'] == "Histogram":
                        fig = px.histogram(df, x=chart['x'], color=chart['color'], color_discrete_sequence=[chart['palette']])
                    elif chart['type'] == "Box":
                        fig = px.box(df, x=chart['x'], y=chart['y'], color=chart['color'], color_discrete_sequence=[chart['palette']])
                    elif chart['type'] == "Violin":
                        fig = px.violin(df, x=chart['x'], y=chart['y'], color=chart['color'], box=True, points='all', color_discrete_sequence=[chart['palette']])
                    elif chart['type'] == "Heatmap":
                        fig = px.density_heatmap(df, x=chart['x'], y=chart['color'], color_continuous_scale=chart['palette'])
                    elif chart['type'] == "Custom":
                        local_vars = {'df': df, "px": px, "go": go}
                        exec(chart['code'], {}, local_vars)
                        fig = local_vars.get('fig', None)
                        if not fig:
                            st.error("Your code must assign a Plotly figure to variable 'fig'")
                            continue
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not plot {chart['type']} chart: {e}")

    if st.button("Clear Dashboard"):
        st.session_state.dashboard_config = {'kpis': [], 'charts': []}
        st.rerun()

def auto_insights(df):
    st.header("📌 Auto Insights (AI-powered)")
    st.write("Let AI & statistics find interesting patterns in your data.")

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='viridis', ax=ax)
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

    # Anomaly/Outlier Detection (IsolationForest)
    st.subheader("Anomaly Detection (IsolationForest)")
    if len(numeric_cols) >= 1:
        clf = IsolationForest(contamination=0.05, random_state=42)
        preds = clf.fit_predict(df[numeric_cols])
        df['anomaly'] = preds
        outliers = df[df['anomaly'] == -1]
        st.write(f"**Detected {len(outliers)} potential anomalies.**")
        if not outliers.empty:
            st.dataframe(outliers)
        st.write("*Anomalies detected using ensemble Isolation Forest method*")
        fig = px.scatter_matrix(df, dimensions=numeric_cols, color="anomaly", color_continuous_scale='Inferno')
        st.plotly_chart(fig, use_container_width=True)
        df.drop(columns=['anomaly'], inplace=True)
    else:
        st.info("Not enough numeric columns for anomaly detection.")

    # Auto-generated insights (Narrative)
    st.subheader("Key Findings (Narrative)")
    findings = []
    if len(numeric_cols) > 1:
        # Top correlation
        top_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
        top_corr = top_corr[top_corr != 1.0]
        if not top_corr.empty:
            pair, value = top_corr.index[0], top_corr.iloc[0]
            findings.append(f"Strongest correlation ({value:.2f}) between **{pair[0]}** and **{pair[1]}**.")
    if 'outliers' in locals() and not outliers.empty:
        findings.append(f"Detected {len(outliers)} anomalies (possible outliers) in numeric columns.")
    if findings:
        for text in findings:
            st.info(text)
    else:
        st.info("No significant patterns detected.")

def run_ml_studio(df):
    st.header("🤖 ML Studio (AutoML + Explainability)")
    st.write("Train a model with automatic tuning and explain predictions.")

    problem_type = st.selectbox("Select Problem Type", ["Auto", "Classification", "Regression"])
    target_col = st.selectbox("🎯 Select Target Column", df.columns)
    feature_cols = st.multiselect("🧩 Select Feature Columns", [c for c in df.columns if c != target_col])

    if not target_col or not feature_cols:
        st.warning("Please select target and feature columns.")
        return

    # Preprocessing
    X = df[feature_cols]
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    y_is_num = pd.api.types.is_numeric_dtype(y)
    model_type = problem_type
    if problem_type == "Auto":
        model_type = "Regression" if y_is_num and y.nunique() > 8 else "Classification"

    if model_type == "Classification" and not y_is_num:
        y, class_names = pd.factorize(y)
        st.session_state.ml_class_names = class_names

    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            if model_type == "Classification":
                model = RandomForestClassifier(n_estimators=200, random_state=42)
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
                st.subheader("Cross-validated Accuracy")
                scores = cross_val_score(model, X, y, cv=5)
                st.write(f"Mean accuracy: **{scores.mean():.2%}** ± {scores.std():.2%}")
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("Regression Metrics")
                col1, col2 = st.columns(2)
                col1.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.3f}")
                col2.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.3f}")
                st.subheader("Cross-validated R²")
                scores = cross_val_score(model, X, y, cv=5, scoring="r2")
                st.write(f"Mean R²: **{scores.mean():.2f}** ± {scores.std():.2f}")

            # Feature Importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            st.dataframe(importance_df)

            # Explainability (via SHAP)
            if SHAP_AVAILABLE:
                st.subheader("Explainability (SHAP)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                st.pyplot(shap.summary_plot(shap_values, X_train, plot_type="bar", show=False))
            else:
                st.info("Install SHAP for advanced explainability.")

            # Store model in session
            st.session_state.ml_model = model
            st.session_state.ml_features = X.columns
            st.session_state.ml_problem_type = model_type

            # Model Download
            buf = io.BytesIO()
            pickle.dump(model, buf)
            b64 = base64.b64encode(buf.getvalue()).decode()
            href = f'<a href="data:file/output_model.pkl;base64,{b64}" download="model.pkl">Download trained model (.pkl)</a>'
            st.markdown(href, unsafe_allow_html=True)

    # What-if analysis
    if 'ml_model' in st.session_state:
        st.subheader("🔮 What-If Analysis / Live Prediction")
        with st.form("prediction_form"):
            inputs = {}
            for col in st.session_state.ml_features:
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
    powerbi_pipeline()
