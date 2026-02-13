🎙️ VoiceInsight
AI Call Center Analysis Engine

VoiceInsight is an end-to-end AI-powered call center analytics system that transcribes customer calls, analyzes sentiment and emotion, extracts key topics, and classifies call types using both machine learning and rule-based logic for comparison and explainability.

🚀 Features

🎧 Audio Upload & Processing

📝 Automatic Speech-to-Text Transcription (Whisper)

🧠 Sentiment Analysis

😊 Emotion Detection

🔑 Keyword Extraction

📞 Call Type Prediction

ML-based (XGBoost trained on Kaggle dataset)

Rule-based (heuristic validation)

🔍 ML vs Rule-Based Comparison

📊 Explainable, Human-Friendly UI

⬇️ Downloadable JSON Analysis Report

🏗️ System Architecture
Audio File
   ↓
Speech-to-Text (Whisper)
   ↓
NLP Analysis
(Sentiment, Emotion, Keywords)
   ↓
Feature Engineering
   ↓
┌─────────────────┬──────────────────┐
│ ML Model (XGB)  │ Rule-Based Engine │
└─────────────────┴──────────────────┘
   ↓
Prediction Comparison
   ↓
Streamlit Frontend

🧠 Why Dual Prediction (ML + Rules)?

In real-world AI systems, relying on a single model is risky.

VoiceInsight uses:

ML Model → adaptable, data-driven predictions

Rule Engine → interpretable, deterministic validation

This allows:

Prediction comparison

Error analysis

Increased trust & explainability

Safer deployment in production-like settings

Project Structure:

VoiceInsight/
│
├── backend/
│   ├── main.py               # FastAPI entry point (/transcribe)
│   ├── predictor.py          # ML + rule-based call type prediction
│   ├── feature_builder.py    # Converts NLP outputs → ML features
│   ├── sentiment.py          # Sentiment analysis logic
│   ├── emotion.py            # Emotion detection logic
│   ├── keywords.py           # Keyword extraction
│   ├── logger.py             # Logs features + predictions to CSV/DB
│
├── frontend/
│   └── app.py                # Streamlit UI
│
├── training/
│   ├── train_call_classifier.py  # Initial model training
│   ├── retrain_from_kaggle.py    # Retraining using Kaggle dataset
│   ├── call_dataset.csv          # Training dataset
│   └── customer_call_transcriptions.csv
│
├── models/
│   ├── call_classifier.pkl       # Trained XGBoost model
│   └── label_encoder.pkl         # Label encoder for classes
│
├── data/
│   ├── audio/                    # Uploaded audio files
│   └── call_features.csv         # Logged features for future retraining
│
├── voiceinsight.db               # SQLite DB (future use / logging)
├── requirements.txt
├── README.md
└── .gitignore


⚙️ Tech Stack

Backend

FastAPI

Python

Whisper

Transformers

XGBoost

Scikit-learn

Frontend

Streamlit

ML & NLP

Sentiment Analysis (Transformer models)

Emotion Classification

Keyword Extraction

Feature Engineering

XGBoost Classifier

🧪 Call Type Classification
ML-Based Prediction

Trained on Kaggle dataset

Uses engineered numerical features:

Sentiment score

Emotion indicators

Keyword count

Transcript length

Polarity flags

Rule-Based Prediction

Keyword heuristics

Sentiment thresholds

Emotion flags

Output Example
{
  "ml_prediction": "customer",
  "rule_prediction": "customer"
}


Agreement → High confidence
Disagreement → Review / retraining candidate

🖥️ Frontend Highlights

Clean dashboard-style UI

Side-by-side sentiment & emotion metrics

Clear call type labeling

AI decision explanation

Downloadable analysis report

Designed for non-technical stakeholders

▶️ How to Run
1️⃣ Backend
cd backend
uvicorn main:app --reload

2️⃣ Frontend
cd frontend
streamlit run app.py

📌 Use Cases

Call center quality monitoring

Customer experience analysis

Sales vs support call identification

Emotion-aware customer handling

AI-assisted call review systems

🔮 Future Improvements

Real-time streaming analysis

Speaker diarization

Model confidence scoring

Active learning from misclassifications

Dashboard analytics (daily trends, alerts)

🎯 Key Learning Outcomes

End-to-end ML system design

Feature engineering for NLP pipelines

Hybrid AI (ML + rules) architecture

Model explainability

Production-oriented API & UI integration

👤 Author

Riddhi Rajgarhia
AI / ML Project – VoiceInsight
