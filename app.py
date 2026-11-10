from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ANN Salary Regression",
    page_icon="📊",
    layout="centered"
)


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")

    model = tf.keras.models.load_model(
        BASE_DIR / "salary_ann.keras"
    )

    with open(BASE_DIR / "metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)

    return preprocessor, model, metadata


st.title("Estimated Salary Prediction")

st.caption("Artificial Neural Network | Regression classroom project")

st.write(
    "Enter customer details to estimate the dataset's EstimatedSalary value. "
    "This demonstration is for learning only and must not be used for financial decisions."
)

required_files = [
    BASE_DIR / "preprocessor.joblib",
    BASE_DIR / "salary_ann.keras",
    BASE_DIR / "metadata.json"
]

missing_files = [
    path.name for path in required_files if not path.is_file()
]

if missing_files:
    st.error(f"Missing model files: {', '.join(missing_files)}")
    st.stop()

try:
    preprocessor, model, metadata = load_artifacts()

except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()


with st.form("salary_input"):

    st.subheader("Customer information")

    c1, c2 = st.columns(2)

    with c1:
        credit_score = st.number_input(
            "Credit score", 300, 900, 650
        )

        geography = st.selectbox(
            "Geography", ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        age = st.number_input(
            "Age", 18, 100, 35
        )

        tenure = st.number_input(
            "Tenure (years)", 0, 10, 5
        )

    with c2:
        balance = st.number_input(
            "Account balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        products = st.selectbox(
            "Number of products", [1, 2, 3, 4]
        )

        has_card = st.selectbox(
            "Has credit card?", ["Yes", "No"]
        )

        active = st.selectbox(
            "Active member?", ["Yes", "No"]
        )

        exited = st.selectbox(
            "Exited the bank?", ["No", "Yes"]
        )

    submitted = st.form_submit_button(
        "Predict estimated salary",
        type="primary"
    )


if submitted:

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active == "Yes" else 0,
        "Exited": 1 if exited == "Yes" else 0
    }])

    expected_columns = metadata["feature_columns"]

    customer = customer.reindex(columns=expected_columns)

    transformed = preprocessor.transform(customer).astype("float32")

    prediction = float(
        model.predict(transformed, verbose=0).reshape(-1)[0]
    )

    st.success("Prediction completed")

    st.metric(
        "Predicted Estimated Salary",
        f"{prediction:,.2f}"
    )

    st.caption(
        "The output is a model estimate in the dataset's currency units, "
        "not a verified salary."
    )


with st.expander("Model evaluation (held-out test set)"):

    metrics = metadata["metrics"]

    st.write(f"**MAE:** {metrics['mae']:,.2f}")
    st.write(f"**RMSE:** {metrics['rmse']:,.2f}")
    st.write(f"**R²:** {metrics['r2']:.4f}")