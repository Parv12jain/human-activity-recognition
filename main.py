# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model

# import os

# app = FastAPI(title="Human Activity Recognition API")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = load_model(os.path.join(BASE_DIR, "har_ann_model.h5"))

# mapping = {
#     0:'Walking',
#     1:"Walking Upstairs",
#     2:"Walking Downstaris",
#     3:"Sitting",
#     4:"Standing",
#     5:"Laying"
# }

# class MODELInput(BaseModel):
#     features: list[float]

# # Endpoint
# @app.post('/predict')
# def predict(data: MODELInput):
#     if len(data.features) != 561:
#         return("error!! Input should have 561 features")

#     input_array = np.array(data.features).reshape(1,-1)

#     prediction = model.predict(input_array)
#     predicted_class = int(np.argmax(prediction))
#     confidence = float(np.max(prediction)* 100)
#     return{
#         "predicted_activity":activity_map[predicted_class],
#         "confidence":round(confidence,2)

#     }

# @app.get("/")
# def Home():
#     return {"message": "API is Live"}


from fastapi import FastAPI, HTTPException, UploadFile, File
import mysql.connector
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import os

# ================= APP INIT =================
app = FastAPI(title="Human Activity Recognition API 🚀")

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "har_ann_model.h5"))

activity_map = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

# ================= MYSQL CONNECTION =================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Parv@123",  # CHANGE THIS
    database="har_system"
)
cursor = db.cursor()

# ================= ROOT =================
@app.get("/")
def home():
    return {"message": "HAR API is Live 🚀 Visit /docs"}

# ================= JSON PREDICTION =================
@app.post("/predict")
def predict(features: list[float]):

    if len(features) != 561:
        raise HTTPException(status_code=400, detail="Need exactly 561 features")

    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    activity = activity_map[predicted_class]

    # Save to database
    cursor.execute(
        "INSERT INTO predictions (activity, confidence, timestamp) VALUES (%s, %s, %s)",
        (activity, confidence, datetime.now())
    )
    db.commit()

    return {
        "predicted_activity": activity,
        "confidence": round(confidence, 2)
    }

# ================= CSV UPLOAD =================
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")

    df = pd.read_csv(file.file)

    if "Activity" in df.columns:
        df = df.drop("Activity", axis=1)

    if df.shape[1] != 561:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain exactly 561 columns. Found {df.shape[1]}"
        )

    data = df.values

    prediction = model.predict(data)
    predicted_classes = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1) * 100

    results = []

    for i in range(len(predicted_classes)):
        activity = activity_map[int(predicted_classes[i])]
        conf = float(confidence[i])

        cursor.execute(
            "INSERT INTO predictions (activity, confidence, timestamp) VALUES (%s, %s, %s)",
            (activity, conf, datetime.now())
        )
        db.commit()

        results.append({
            "sample": i,
            "predicted_activity": activity,
            "confidence": round(conf, 2)
        })

    return {
        "total_samples": len(results),
        "predictions": results
    }

# ================= HISTORY =================
@app.get("/history")
def get_history():
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    return {"history": rows}
