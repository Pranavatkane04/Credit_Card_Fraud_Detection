import streamlit as st
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# ✅ Load the saved model
model_path = "credit_card_best_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Error: Model file '{model_path}' not found! Please check the file path.")
    st.stop()

# ✅ Function to load Lottie animations safely
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None

# ✅ Load animations (check for errors)
lottie_fraud = load_lottie_url("https://lottie.host/0421465d-9bfb-4df4-91f2-54a272d7e3ae/fraud.json")
lottie_secure = load_lottie_url("https://lottie.host/3a8c7a36-4c29-44b2-8481-04554b5d5709/secure.json")

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

# ✅ UI Header
st.markdown("## 💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details or use auto-fill buttons to analyze fraud probability.")

# ✅ Initialize Session State for Auto-Fill
if "input_values" not in st.session_state:
    st.session_state.input_values = {
        "Time": 0.0, "Amount": 0.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }

# ✅ Create Input Layout with Tabs
tabs = st.tabs(["Transaction Info", "Feature Inputs (V1 - V14)", "Feature Inputs (V15 - V28)"])

# ✅ Transaction Info Tab
with tabs[0]:
    st.markdown("### ⏳ Transaction Details")
    st.session_state.input_values["Time"] = st.number_input("Transaction Time", value=st.session_state.input_values["Time"], format="%.8f")
    st.session_state.input_values["Amount"] = st.number_input("Transaction Amount", value=st.session_state.input_values["Amount"], format="%.8f")

# ✅ Feature Inputs Tab 1 (V1 - V14)
with tabs[1]:
    st.markdown("### Feature Variables (V1 - V14)")
    cols = st.columns(2)
    for i in range(1, 15):
        with cols[0] if i <= 7 else cols[1]:
            st.session_state.input_values[f"V{i}"] = st.number_input(f"V{i}", value=st.session_state.input_values[f"V{i}"], format="%.8f")

# ✅ Feature Inputs Tab 2 (V15 - V28)
with tabs[2]:
    st.markdown("### Feature Variables (V15 - V28)")
    cols = st.columns(2)
    for i in range(15, 29):
        with cols[0] if i <= 21 else cols[1]:
            st.session_state.input_values[f"V{i}"] = st.number_input(f"V{i}", value=st.session_state.input_values[f"V{i}"], format="%.8f")

# ✅ Auto-Fill Buttons (Use Session State)
col1, col2 = st.columns(2)
if col1.button("🔵 Fill Genuine Data"):
    st.session_state.input_values.update({
        "Time": 32456, "Amount": 234.56,
        "V1": -3.245, "V2": 1.567, "V3": -0.789, "V4": 2.123, "V5": -1.234, "V6": 0.456, "V7": -1.678,
        "V8": 0.234, "V9": -0.567, "V10": 0.678, "V11": -0.123, "V12": 0.456, "V13": -0.789, "V14": 0.912,
        "V15": -0.345, "V16": 0.678, "V17": -1.123, "V18": 0.789, "V19": -0.234, "V20": 1.456, "V21": -1.567,
        "V22": 0.789, "V23": -0.678, "V24": 0.123, "V25": 0.456, "V26": 0.345, "V27": -0.234, "V28": 1.567
    })
    st.rerun()

if col2.button("🔴 Fill Fraud Data"):
    st.session_state.input_values.update({
        "Time": 4462, "Amount": 239.93,
        "V1": -2.30335, "V2": 1.75924, "V3": -0.35974, "V4": 2.33024, "V5": -0.82163, "V6": -0.07579, "V7": 0.56232,
        "V8": -0.39915, "V9": -0.23825, "V10": -1.52541, "V11": 2.03291, "V12": -6.56012, "V13": 0.02294, "V14": -1.47010,
        "V15": -0.69883, "V16": -2.28219, "V17": -4.78183, "V18": -2.61566, "V19": -1.33444, "V20": -0.43002, "V21": -0.29417,
        "V22": -0.93239, "V23": 0.17273, "V24": -0.08733, "V25": -0.15611, "V26": -0.54263, "V27": 0.03957, "V28": -0.15303
    })
    st.rerun()

# ✅ Predict Button
if st.button("🚀 Predict Fraud"):
    user_input = np.array([[st.session_state.input_values["Time"]] + 
                           [st.session_state.input_values[f"V{i}"] for i in range(1, 29)] + 
                           [st.session_state.input_values["Amount"]]])

    prediction = model.predict(user_input)
    proba = model.predict_proba(user_input)
    st.write(f"🔍 Probability Output: {proba}")

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
        if lottie_fraud:
            st_lottie(lottie_fraud, height=200)
    else:
        st.success("✅ Genuine Transaction")
        if lottie_secure:
            st_lottie(lottie_secure, height=200)
