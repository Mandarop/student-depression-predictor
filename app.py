from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_enc.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data from request

        # Extract features
        features = [
            float(data["Academic Pressure"]),
            float(data["Work Pressure"]),
            float(data["Study Satisfaction"]),
            data["Sleep Duration"],  # Categorical
            float(data["Financial Stress"]),
        ]

        # Encode Sleep Duration
        features[3] = label_encoders["Sleep Duration"].transform([features[3]])[0]

        # Scale the input
        scaled_features = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled_features)[0]
        risk = "HIGH RISK" if prediction == 1 else "LOW RISK"

        return jsonify({"prediction": risk})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
