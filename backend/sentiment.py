from transformers import pipeline

# Load sentiment analysis pipeline
# This uses a pre-trained BERT-based model
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text: str):
    """
    Takes text as input and returns sentiment label and confidence score
    """
    result = sentiment_analyzer(text)[0]

    return {
        "label": result["label"],      # POSITIVE / NEGATIVE
        "score": result["score"]        # Confidence score
    }
