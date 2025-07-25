import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import LabelEncoder

def run_modeling(df):
    st.subheader("🧠 Self-Training Model Builder")

    # Target selection
    target = st.selectbox("Select Target Column", df.columns)
    if target:
        features = df.drop(columns=[target])
        X = pd.get_dummies(features)

        # Label encode the target
        y_raw = df[target]
        y = LabelEncoder().fit_transform(y_raw)

        # Split dataset (simulate unlabeled data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        # Simulate partial labels
        import numpy as np
        rng = np.random.RandomState(42)
        mask = rng.rand(len(y_train)) < 0.5
        y_train_partial = np.copy(y_train)
        y_train_partial[~mask] = -1  # Unlabeled

        # Base model
        base_model = RandomForestClassifier(n_estimators=100)
        self_training_model = SelfTrainingClassifier(base_model)

        # Train
        self_training_model.fit(X_train, y_train_partial)

        # Predict
        y_pred = self_training_model.predict(X_test)

        # Output
        st.markdown("### ✅ Model Performance")
        st.code(classification_report(y_test, y_pred), language='text')
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
