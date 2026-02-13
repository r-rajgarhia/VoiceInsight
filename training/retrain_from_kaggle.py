import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 1. Load Kaggle dataset
df = pd.read_csv("customer_call_transcriptions.csv")

# 2. Convert sentiment → call_type
label_map = {
    "negative": "bad",
    "neutral": "customer",
    "positive": "sales"
}

df["call_type"] = df["sentiment_label"].map(label_map)

# 3. VERY SIMPLE FEATURES (no NLP yet)
X = []
y = []

for text, label in zip(df["text"], df["call_type"]):
    features = [
        len(text),                 # transcript length
        text.count("!"),           # emotional intensity
        text.count("?")            # customer intent
    ]
    X.append(features)
    y.append(label)

# 4. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# save encoder
joblib.dump(label_encoder, "../models/label_encoder.pkl")

# -----------------------------
# 5. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Train model
# -----------------------------
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=3
)

model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -----------------------------
# 8. Save model
# -----------------------------
joblib.dump(model, "../models/call_classifier.pkl")

print("✅ Model training completed successfully")
