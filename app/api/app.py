from flask import Flask, request, jsonify
import joblib
import os
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
@app.route("/")
def home():
    return {"message": "Matchify Resume Screening API is running!"}
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "resume_text" not in data:
        return jsonify({"error": "Missing resume_text"}), 400
    resume_text = data["resume_text"]
    X = vectorizer.transform([resume_text])
    prediction = model.predict(X)[0]
    return jsonify({"predicted_job_role": prediction})
=======

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

>>>>>>> 872d4b79e0adf2d4f3640631de5dc2deb0d483f8
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
