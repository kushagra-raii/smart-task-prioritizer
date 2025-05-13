from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow all origins by default

# Load the trained model and preprocessing components
model = joblib.load("task_prioritizer_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
features_to_scale = joblib.load("features_to_scale.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        
        # Extract features
        importance = float(data['Importance'])
        deadline_days = float(data['Deadline_Days'])
        task_status = data['Task_Status']
        number_of_dependents = int(data['Number_of_Dependents'])

        # Preprocess input
        # Encode task status
        task_status_encoded = encoders['Task_Status'].transform([task_status])[0]
        
        # Scale numerical features
        scaled_features = scaler.transform([[importance, deadline_days, number_of_dependents]])
        
        # Combine all features
        features = [
            scaled_features[0][0],  # Scaled Importance
            scaled_features[0][1],  # Scaled Deadline_Days
            task_status_encoded,
            scaled_features[0][2]   # Scaled Number_of_Dependents
        ]

        # Make prediction
        priority = model.predict([features])[0]

        # Make sure your response includes proper headers
        response = jsonify({
            'priority': float(priority),
            'message': 'Priority calculated successfully'
        })
        return response

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)