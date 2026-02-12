import joblib
import numpy as np

# Load trained model and label encoder ONCE
model = joblib.load("models/call_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def predict_call_type(features: dict):
    """
    Takes engineered features and returns predicted call label
    """

    # Convert feature dict into model input order
    X = np.array([[
        features["sentiment_score"],
        features["neutral"],
        features["angry"],
        features["happy"],
        features["keywords_count"],
        features["transcript_length"]
    ]])

    # Predict class index
    pred_class = model.predict(X)[0]

    # Convert numeric label back to string
    label = label_encoder.inverse_transform([pred_class])[0]

    return label
