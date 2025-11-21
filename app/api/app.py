from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Absolute paths inside the container
MODEL_PATH = "app/model/model.pkl"
VECTORIZER_PATH = "app/model/vectorizer.pkl"

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/")
def home():
    return {"message": "Matchify Resume Screening API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if "resume_text" not in data:
        return jsonify({"error": "Missing resume_text"}), 400

    resume_text = data["resume_text"]

    # Transform text
    X = vectorizer.transform([resume_text])

    # Predict job role
    prediction = model.predict(X)[0]

    return jsonify({
        "predicted_job_role": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
