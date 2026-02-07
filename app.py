import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import mysql.connector
from datetime import datetime

# ---------------- Page Config ----------------
st.set_page_config(page_title="HAR AI System", layout="wide")

# ---------------- Load Model ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ann_model = load_model(os.path.join(BASE_DIR, "har_ann_model.h5"))

# ---------------- MySQL Connection ----------------
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Parv@123",
        database="har_system"
    )
    cursor = db.cursor()
    db_connected = True
except:
    db_connected = False

# ---------------- Activity Mapping ----------------
activity_map = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

# ---------------- Sidebar ----------------
st.sidebar.header("📊 Model Performance")
st.sidebar.write("Model: Artificial Neural Network (ANN)")
st.sidebar.write("Dataset: UCI HAR (561 engineered features)")
st.sidebar.write("Test Accuracy: ~94–97%")
st.sidebar.write("Classes: 6 Activities")

if db_connected:
    st.sidebar.success("✅ MySQL Connected")
else:
    st.sidebar.error("❌ MySQL Not Connected")

with st.sidebar.expander("ℹ️ About the Model"):
    st.write("""
    This model is trained on 561 engineered features extracted from 
    smartphone accelerometer and gyroscope signals.
    
    The ANN learns patterns from time-domain and frequency-domain 
    features to classify human activities into six categories.
    
    The output layer uses Softmax activation to produce probability scores.
    """)

# ---------------- UI ----------------
st.title("📱 Human Activity Recognition System")
st.markdown("### AI-powered Activity Classification Dashboard")

uploaded_file = st.file_uploader("Upload CSV (561 feature columns)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "Activity" in df.columns:
            df = df.drop("Activity", axis=1)

        df = df.select_dtypes(include=['number'])

        st.write("### 📊 Dataset Info")
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        if df.shape[1] != 561:
            st.error(f"CSV must contain exactly 561 feature columns. Found {df.shape[1]}.")
        else:
            data = df.values

            prediction = ann_model.predict(data)
            predicted_class = np.argmax(prediction, axis=1)
            confidence = np.max(prediction, axis=1) * 100

            # ---------------- Row Selection ----------------
            st.write("---")
            st.subheader("🔎 Select Sample to View Prediction")

            row_index = st.slider(
                "Choose Sample Index",
                min_value=0,
                max_value=len(data)-1,
                value=0
            )

            selected_activity = activity_map[predicted_class[row_index]]
            selected_confidence = float(confidence[row_index])

            # Show Prediction
            st.success(f"Predicted Activity: {selected_activity}")
            st.progress(int(selected_confidence))
            st.info(f"Confidence: {selected_confidence:.2f}%")

            # ---------------- Save to MySQL ----------------
            if db_connected:
                cursor.execute(
                    "INSERT INTO predictions (activity, confidence, timestamp) VALUES (%s, %s, %s)",
                    (selected_activity, selected_confidence, datetime.now())
                )
                db.commit()

            # ---------------- Probability Chart ----------------
            st.write("### 📈 Class Probabilities")

            prob_df = pd.DataFrame({
                "Activity": list(activity_map.values()),
                "Probability (%)": prediction[row_index] * 100
            })

            st.bar_chart(prob_df.set_index("Activity"))

            # ---------------- Simulation ----------------
            st.write("### ▶ Simulate Real-Time Prediction")

            if st.button("Start Simulation"):
                for i in range(min(10, len(data))):
                    st.write(f"Sample {i} → {activity_map[predicted_class[i]]} ({confidence[i]:.2f}%)")

            # ---------------- All Predictions Table ----------------
            st.write("### 📋 All Predictions")

            results_df = pd.DataFrame({
                "Predicted Class": predicted_class,
                "Predicted Activity": [activity_map[i] for i in predicted_class],
                "Confidence (%)": confidence
            })

            st.dataframe(results_df)

            # ---------------- Summary ----------------
            st.write("### 📌 Prediction Summary")
            summary = results_df["Predicted Activity"].value_counts()
            st.bar_chart(summary)

            # ---------------- Download Results ----------------
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name="har_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------------- View MySQL History ----------------
st.write("---")
st.write("### 🗄 Prediction History (MySQL Database)")

if db_connected and st.button("Load History from Database"):
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    history_df = pd.DataFrame(rows, columns=["ID", "Activity", "Confidence", "Timestamp"])
    st.dataframe(history_df)

    st.write("### 📊 Database Activity Summary")
    summary_db = history_df["Activity"].value_counts()
    st.bar_chart(summary_db)
