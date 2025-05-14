from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# App initialization
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML components
model = joblib.load("task_prioritizer_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
features_to_scale = joblib.load("features_to_scale.pkl")  # Not used directly

# Input schema
class TaskData(BaseModel):
    Importance: float
    Deadline_Days: float
    Task_Status: str
    Number_of_Dependents: int

# Endpoint
@app.post("/predict")
def predict(data: TaskData):
    try:
        # Encode task status
        task_status_encoded = encoders['Task_Status'].transform([data.Task_Status])[0]

        # Scale numerical features
        scaled_features = scaler.transform([[data.Importance, data.Deadline_Days, data.Number_of_Dependents]])

        # Final feature vector
        features = [
            scaled_features[0][0],  # Importance
            scaled_features[0][1],  # Deadline
            task_status_encoded,    # Encoded status
            scaled_features[0][2],  # Dependents
        ]

        # Predict
        priority = model.predict([features])[0]

        return {
            "priority": float(priority),
            "message": "Priority calculated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

