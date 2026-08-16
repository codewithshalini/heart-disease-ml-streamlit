import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Page Setup
st.set_page_config(
    page_title="Heart Disease ML Evaluation & Diagnostic Dashboard",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 25px; }
    .status-positive { color: #DC2626; font-weight: bold; }
    .status-negative { color: #16A34A; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">❤️ Heart Disease Model Evaluation & Patient Diagnosis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload test_data.csv to generate patient diagnostic predictions and evaluate model classification metrics.</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration Controls")
uploaded_file = st.sidebar.file_uploader("1. Upload 'test_data.csv'", type=["csv"])

model_choice = st.sidebar.selectbox(
    "2. Choose ML Model Architecture",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbor",
        "Naive Bayes",
        "Random Forest (Ensemble)"
    ]
)

@st.cache_resource
def load_model_assets():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "K-Nearest Neighbor": joblib.load("model/k-nearest_neighbor.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest (Ensemble)": joblib.load("model/random_forest_ensemble.pkl")
    }
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler

models, scaler = load_model_assets()
selected_model = models[model_choice]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Identify target column name safely
    target_col = 'target' if 'target' in df.columns else ('target_binary' if 'target_binary' in df.columns else None)

    if target_col is None:
        st.error("Uploaded CSV file must contain a 'target' or 'target_binary' ground truth column.")
    else:
        # Separate features and target
        X_test = df.drop(columns=[col for col in ['target', 'target_binary', 'num'] if col in df.columns])
        y_test = df[target_col]

        # Apply Feature Scaling if required by model type
        if model_choice in ["Logistic Regression", "K-Nearest Neighbor", "Naive Bayes"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        # Generate Model Predictions & Risk Probabilities
        y_pred = selected_model.predict(X_eval)
        y_proba = selected_model.predict_proba(X_eval)[:, 1]

        # Calculate Benchmark Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # ----------------------------------------------------
        # SECTION 1: Benchmark Performance Metrics
        # ----------------------------------------------------
        st.subheader(f"📊 Benchmark Performance Metrics: {model_choice}")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC Score", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC Score", f"{mcc:.4f}")

        st.divider()

        # ----------------------------------------------------
        # SECTION 2: Patient Diagnosis Output Table
        # ----------------------------------------------------
        st.subheader("📋 Patient Diagnostic Predictions Summary Table")
        st.write("This table displays the model's explicit diagnosis outcome and calculated risk percentage for each patient in the uploaded test set:")

        # Construct decorated results dataframe
        results_df = df.copy()
        results_df['Predicted_Diagnosis'] = ["⚠️ Disease Detected" if p == 1 else "✅ Healthy / No Disease" for p in y_pred]
        results_df['Heart_Disease_Risk (%)'] = np.round(y_proba * 100, 2)
        results_df['Actual_Ground_Truth'] = ["Disease" if t == 1 else "Healthy" for t in y_test]

        # Select primary key columns to present clearly
        display_cols = ['Predicted_Diagnosis', 'Heart_Disease_Risk (%)', 'Actual_Ground_Truth'] + [c for c in ['age', 'sex', 'cp', 'trestbps', 'chol'] if c in df.columns]

        st.dataframe(
    results_df[display_cols].style.map(
        lambda val: 'background-color: #FEE2E2; font-weight: bold;' if val == "⚠️ Disease Detected" else ('background-color: #D1FAE5; font-weight: bold;' if val == "✅ Healthy / No Disease" else ''),
        subset=['Predicted_Diagnosis']
    ),
    use_container_width=True
)

        # Dataset Diagnostic Breakdown Summary
        total_patients = len(y_pred)
        disease_cases = int(np.sum(y_pred == 1))
        healthy_cases = total_patients - disease_cases

        st.info(f"💡 **Batch Diagnosis Breakdown:** Out of **{total_patients}** evaluated test patients, **{disease_cases} patients ({disease_cases/total_patients*100:.1f}%)** are predicted to have **Heart Disease**, while **{healthy_cases} patients ({healthy_cases/total_patients*100:.1f}%)** are diagnosed as **Healthy**.")

        st.divider()

        # ----------------------------------------------------
        # SECTION 3: Diagnostic Visualizations
        # ----------------------------------------------------
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Healthy", "Disease"],
                        yticklabels=["Healthy", "Disease"])
            ax.set_xlabel("Predicted Diagnosis")
            ax.set_ylabel("Actual Diagnosis")
            st.pyplot(fig)

        with c2:
            st.subheader("📜 Detailed Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))

else:
    st.info("👈 Upload `test_data.csv` using the sidebar control to view patient predictions and evaluation metrics.")
