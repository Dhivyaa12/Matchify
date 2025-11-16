import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# TRAINING DATA (TEMPORARY)
# -----------------------------
data = {
    "resume_text": [
        "Python, Machine Learning, Data Science, Pandas",
        "Java, Spring Boot, Microservices, MySQL",
        "JavaScript, React, Node.js, HTML, CSS",
        "AWS, Terraform, Docker, Jenkins, DevOps",
    ],
    "job_role": [
        "ML Engineer",
        "Backend Developer",
        "Frontend Developer",
        "DevOps Engineer",
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# TEXT VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume_text"])
y = df["job_role"]

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# SAVE MODEL & VECTORIZER
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training complete. Files saved: model.pkl & vectorizer.pkl")
