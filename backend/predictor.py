import joblib
import numpy as np

# Load trained model and label encoder ONCE
model = joblib.load("models/call_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def predict_call_type(features: dict) -> str:
    sentiment = features.get("sentiment")
    emotion = features.get("emotion")
    keywords = features.get("keywords", [])
    transcript = features.get("transcript", "").lower()

    # Sales call
    if any(k in transcript for k in ["buy", "purchase", "price", "plan", "upgrade"]):
        return "sales"

    # Bad / angry call
    if emotion in ["angry", "frustrated"] or sentiment == "negative":
        return "bad"

    # Default
    return "customer"

def predict_call_type_from_text(transcript: str):
    """
    Predict call type from transcript text
    """

    # SAME features used during training
    features = np.array([[
        len(transcript),
        transcript.count("!"),
        transcript.count("?")
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    call_type = label_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))

    return call_type, confidence
