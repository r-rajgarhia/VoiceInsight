from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os

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

    return {
        "filename": file.filename,
        "language": info.language,
        "transcript": transcript
    }
