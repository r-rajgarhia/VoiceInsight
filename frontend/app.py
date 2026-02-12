import streamlit as st
import requests

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(
    page_title="VoiceInsight",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ VoiceInsight")
st.subheader("AI Call Center Analysis Engine")
st.write("Upload an audio file to transcribe and analyze emotion & sentiment.")

# Backend API URL
API_URL = "http://127.0.0.1:8000/transcribe"

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "ogg"]
)

# -------------------------------
# Transcribe button
# -------------------------------
if uploaded_file is not None:
    if st.button("Transcribe"):
        with st.spinner("Processing audio..."):
            # Send file to FastAPI backend
            files = {
                "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
            }

            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                data = response.json()

                st.success("Transcription successful!")

                # -------------------------------
                # Display results
                # -------------------------------
                st.subheader("📝 Transcript")
                st.write(data["transcript"])

                st.subheader("📊 Sentiment")
                st.write(f"**{data['sentiment']['label']}**")
                st.write(f"Confidence: {data['sentiment']['score']:.2f}")

                st.subheader("😊 Emotion")
                st.write(f"**{data['emotion']['label']}**")
                st.write(f"Confidence: {data['emotion']['score']:.2f}")

                st.subheader("🔑 Keywords")
                st.write(", ".join(data["keywords"]))

            else:
                st.error("Something went wrong. Please try again.")