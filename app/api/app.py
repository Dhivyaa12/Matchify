from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("../model/model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

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
