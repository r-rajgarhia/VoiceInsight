import csv
import os

DATASET_PATH = "data/call_features.csv"

def log_call_features(features: dict, label: str, rule_label: str):
    file_exists = os.path.exists(DATASET_PATH)

    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "sentiment_score",
                "neutral",
                "angry",
                "happy",
                "negative_score",
                "positive_score",
                "is_negative",
                "keywords_count",
                "transcript_length",
                "label"
            ])

        writer.writerow([
            features["sentiment_score"],
            features["neutral"],
            features["angry"],
            features["happy"],
            features["negative_score"],
            features["positive_score"],
            features["is_negative"],
            features["keywords_count"],
            features["transcript_length"],
            label
        ])
