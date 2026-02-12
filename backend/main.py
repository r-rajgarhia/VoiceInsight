from pyexpat import features
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os
import sqlite3
from backend.sentiment import analyze_sentiment
from backend.keywords import extract_keywords
from backend.emotion import analyze_emotion
from backend.feature_builder import build_features
from backend.logger import log_call_features
from backend.predictor import predict_call_type


# Initialize FastAPI application
app = FastAPI(title="VoiceInsight")

# Load speech-to-text model once at startup using Faster-Whisper for better speed and Python 3.13 compatibility
model = WhisperModel("small", device="cpu")

# Directory to store uploaded audio files
UPLOAD_DIR = "data/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create folder if it doesn't exist


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    #Accepts an audio file and returns its text transcription

    # Save uploaded audio file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run speech-to-text on saved audio file
    segments, info = model.transcribe(file_path)

    # Combine all recognized speech segments into one string
    transcript = " ".join(segment.text for segment in segments)

    # Analyze sentiment of the transcribed text
    sentiment = analyze_sentiment(transcript)
    emotion = analyze_emotion(transcript)
    keywords = extract_keywords(transcript)

    # Build features for logging and potential model training
    features = build_features(
    sentiment=sentiment,
    emotion=emotion,
    keywords=keywords,
    transcript=transcript
    )

    # ML prediction
    call_type = predict_call_type(features)
    print("FEATURES:", features)
    log_call_features(
    features,
    label=call_type
    )

    return {
        "filename": file.filename,
        "language": info.language,
        "transcript": transcript,
        "sentiment": sentiment,
        "emotion": emotion,
        "keywords": keywords,
        "predicted_call_type": call_type
    }

def save_call(features: dict, label: str = None):
    conn = sqlite3.connect("voiceinsight.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO calls (
            transcript,
            sentiment_score,
            neutral,
            angry,
            happy,
            keywords_count,
            transcript_length,
            label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        features["transcript"],
        features["sentiment_score"],
        features["neutral"],
        features["angry"],
        features["happy"],
        features["keywords_count"],
        features["transcript_length"],
        label
    ))

    conn.commit()
    conn.close()

@app.get("/health")
def health_check():
    #Health check endpoint to verify server and models are running.
    
    return {
        "status": "ok",
        "message": "VoiceInsight API is running",
        "models": {
            "speech_to_text": "faster-whisper (small)",
            "sentiment": "distilbert-sst2",
            "emotion": "distilroberta-emotion"
        }
    }
