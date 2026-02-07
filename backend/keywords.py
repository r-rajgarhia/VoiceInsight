import spacy

# Load small English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str):
    """
    Extracts important keywords (nouns & proper nouns)
    from the given text using spaCy.
    """

    doc = nlp(text)

    keywords = [
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
    ]

    # Remove duplicates
    return list(set(keywords))
