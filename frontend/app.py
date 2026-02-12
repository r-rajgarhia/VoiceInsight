import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="VoiceInsight",
    page_icon="🎙️",
    layout="wide"
)

# App header
st.title("🎙️ VoiceInsight")
st.subheader("AI Call Center Analysis Engine")
st.write(
    "Upload a call recording to analyze transcription, sentiment, emotion, "
    "keywords, and predict call type using a trained ML model."
)

# Backend API endpoint
API_URL = "http://127.0.0.1:8000/transcribe"

# File uploader
uploaded_file = st.file_uploader(
    "📂 Upload an audio file",
    type=["wav", "mp3", "ogg"]
)

# Transcribe & Analyze
if uploaded_file is not None:
    if st.button("🚀 Analyze Call"):
        with st.spinner("Processing audio with AI models..."):
            files = {
                "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
            }

            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()
            st.success("✅ Analysis completed successfully!")

            # Transcript
            st.subheader("📝 Call Transcript")
            st.write(data["transcript"])

            # AI Analysis (Sentiment & Emotion)
            st.subheader("🧠 AI Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Sentiment",
                    value=data["sentiment"]["label"],
                    delta=f"{data['sentiment']['score']:.2f}"
                )

            with col2:
                st.metric(
                    label="Emotion",
                    value=data["emotion"]["label"],
                    delta=f"{data['emotion']['score']:.2f}"
                )

            # Keywords
            st.subheader("🔑 Key Topics Detected")
            if data["keywords"]:
                st.write(", ".join(data["keywords"]))
            else:
                st.write("No significant keywords detected.")

            # Call Type Prediction (ML Model)
            st.subheader("📞 Call Type Prediction (ML Model)")
            call_type = data["predicted_call_type"].lower()

            if call_type == "sales":
                st.success("💼 Sales Call")
            elif call_type == "support":
                st.warning("🛠 Customer Support Call")
            elif call_type == "bad":
                st.error("🚨 Negative / Problematic Call")
            else:
                st.info(call_type.capitalize())

            # AI Explanation
            st.subheader("🤖 How the AI Reached This Decision")
            st.write(
                f"""
                This call was analyzed using a multi-stage AI pipeline:

                • Speech-to-text transcription (Whisper)  
                • Sentiment analysis (Transformer model)  
                • Emotion detection (Emotion classifier)  
                • Keyword extraction  
                • Feature engineering  
                • Call classification using a trained **XGBoost model**

                Based on these signals, the system classified this call as
                **{call_type.upper()}**.
                """
            )

            # Download analysis
            st.download_button(
                label="⬇️ Download Analysis Report",
                data=json.dumps(data, indent=2),
                file_name="voiceinsight_analysis.json",
                mime="application/json"
            )

        else:
            st.error("❌ Something went wrong while processing the request.")
