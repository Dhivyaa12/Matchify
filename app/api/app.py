from flask import Flask, request, jsonify
import joblib
import os
app = Flask(__name__)

# Candidate paths for model and vectorizer (development and container layouts)
MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "model.pkl"),
    os.path.join("app", "model", "model.pkl"),
]
VECTORIZER_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "vectorizer.pkl"),
    os.path.join("app", "model", "vectorizer.pkl"),
]

def _load_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(f"Could not find any of the model/vectorizer files in: {candidates}")

# Load model and vectorizer from the first existing path
model = _load_first_existing(MODEL_CANDIDATES)
vectorizer = _load_first_existing(VECTORIZER_CANDIDATES)

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
