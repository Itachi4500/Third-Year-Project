import streamlit as st
import pandas as pd

def run_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    analysis_option = st.selectbox("Choose EDA Operation", [
        "Data Overview",
        "Column Types",
        "Summary Statistics",
        "Missing Values",
        "Unique Values",
        "Correlation Matrix",
        "Value Counts for Categorical Columns",
        "Skewness and Kurtosis",
        "Top & Bottom Records"
    ])

    if analysis_option == "Data Overview":
        st.write("### Dataset Preview")
        st.dataframe(df.head())

    elif analysis_option == "Column Types":
        st.write("### Column Data Types")
        st.write(df.dtypes)

    elif analysis_option == "Summary Statistics":
        st.write("### Summary Statistics")
        st.write(df.describe(include='all'))

    elif analysis_option == "Missing Values":
        st.write("### Missing Values Count")
        st.write(df.isnull().sum())
        st.write("### Percentage of Missing Values")
        st.write((df.isnull().sum() / len(df)) * 100)

    elif analysis_option == "Unique Values":
        st.write("### Unique Values in Each Column")
        unique_vals = {col: df[col].nunique() for col in df.columns}
        st.write(pd.DataFrame(unique_vals.items(), columns=["Column", "Unique Values"]))

    elif analysis_option == "Correlation Matrix":
        st.write("### Correlation Matrix")
        corr_matrix = df.corr(numeric_only=True)
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

    elif analysis_option == "Value Counts for Categorical Columns":
        cat_cols = df.select_dtypes(include='object').columns
        if not cat_cols.empty:
            for col in cat_cols:
                st.write(f"### Value Counts for `{col}`")
                st.write(df[col].value_counts())
        else:
            st.warning("No categorical columns found.")

    elif analysis_option == "Skewness and Kurtosis":
        st.write("### Skewness & Kurtosis (Numeric Columns)")
        numeric_cols = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_cols.empty:
            skew_kurt = pd.DataFrame({
                "Skewness": numeric_cols.skew(),
                "Kurtosis": numeric_cols.kurt()
            })
            st.dataframe(skew_kurt)
        else:
            st.warning("No numeric columns found.")

    elif analysis_option == "Top & Bottom Records":
        n = st.slider("Select number of rows", 1, 20, 5)
        st.write("### Top Records")
        st.dataframe(df.head(n))
        st.write("### Bottom Records")
        st.dataframe(df.tail(n))
        
elif choice == "Exploratory Data Analysis":
    run_eda(st.session_state.df)
