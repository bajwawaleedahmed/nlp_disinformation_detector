import json
import spacy
from textblob import TextBlob
import textstat
from tqdm import tqdm
from pathlib import Path
from collections import Counter

# Load spaCy model with full pipeline
nlp = spacy.load("en_core_web_sm")

# Input and output paths
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_preprocessed_articles.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features.json")


def extract_features(article):
    text = article["text"]
    label = article["label"]

    # POS tagging and entity recognition
    doc = nlp(text)
    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = sum(pos_counts.values())

    pos_features = {
        "noun_ratio": pos_counts["NOUN"] / total_tokens if total_tokens else 0,
        "verb_ratio": pos_counts["VERB"] / total_tokens if total_tokens else 0,
        "adj_ratio": pos_counts["ADJ"] / total_tokens if total_tokens else 0,
        "adv_ratio": pos_counts["ADV"] / total_tokens if total_tokens else 0,
    }

    # Correctly use named entities
    entity_counts = Counter([ent.label_ for ent in doc.ents])
    ner_features = {
        "num_ORG": entity_counts["ORG"],
        "num_GPE": entity_counts["GPE"],
        "num_PERSON": entity_counts["PERSON"],
        "num_DATE": entity_counts["DATE"]
    }

    # Sentiment analysis
    blob = TextBlob(text)
    sentiment = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

    # Readability metrics
    readability = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "avg_sentence_length": textstat.avg_sentence_length(text),
        "word_count": textstat.lexicon_count(text)
    }

    return {
        "index": article["index"],
        **pos_features,
        **ner_features,
        **sentiment,
        **readability,
        "label": label
    }


if __name__ == "__main__":
    with input_path.open("r", encoding="utf-8") as infile:
        articles = json.load(infile)

    features = [extract_features(article) for article in tqdm(articles)]

    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(features, outfile, indent=2)

    print(f"✅ Features extracted and saved to {output_path}")
