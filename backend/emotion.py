from transformers import pipeline

# Load emotion classification model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def analyze_emotion(text: str):
    """
    Analyzes emotions in text and returns the most dominant emotion
    with its confidence score.
    """

    results = emotion_classifier(text)[0]

    # Find emotion with highest score
    top_emotion = max(results, key=lambda x: x["score"])

    return {
        "emotion": top_emotion["label"],
        "score": top_emotion["score"]
    }
