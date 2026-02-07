from transformers import pipeline

# Load emotion classification model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def analyze_emotion(text: str):
    results = emotion_classifier(text)

    # If model returns a list, take first element
    if isinstance(results, list):
        results = results[0]

    return {
        "label": results["label"],
        "score": results["score"]
    }
