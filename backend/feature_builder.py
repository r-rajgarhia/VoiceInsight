def build_features(sentiment, emotion, keywords, transcript):
    """
    Converts raw NLP outputs into ML-ready numerical features
    that match the trained model schema.
    """

    # Initialize emotion scores
    neutral = angry = happy = 0.0

    # Assign score to predicted emotion
    if emotion["label"] == "neutral":
        neutral = emotion["score"]
    elif emotion["label"] == "angry":
        angry = emotion["score"]
    elif emotion["label"] == "happy":
        happy = emotion["score"]

    # ---- derived features (THIS IS THE FIX) ----
    negative_score = angry
    positive_score = happy
    is_negative = int(negative_score > positive_score)

    features = {
        "sentiment_score": sentiment["score"],

        "neutral": neutral,
        "angry": angry,
        "happy": happy,

        "negative_score": negative_score,
        "positive_score": positive_score,
        "is_negative": is_negative,

        "keywords_count": len(keywords),
        "transcript_length": len(transcript.split())
    }

    return features
