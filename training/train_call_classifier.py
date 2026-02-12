import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("training/call_dataset.csv")

# Features & target
X = df[
    [
        "sentiment_score",
        "neutral",
        "angry",
        "happy",
        "keywords_count",
        "transcript_length"
    ]
]

y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split (NO stratify — small data friendly)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

# Train
model.fit(X_train, y_train)

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/call_classifier.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("✅ Model trained successfully")
