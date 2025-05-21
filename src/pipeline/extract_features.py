from collections import Counter
from textblob import TextBlob
from pathlib import Path
from tqdm import tqdm
import textstat
import spacy
import json

nlp = spacy.load("en_core_web_sm")

input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/processed/preprocessed_data_with_kb.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/processed/preprocessed_data_with_features.json")


def extract_features(article):
    text = article["text"]
    doc = nlp(text)
    pos_counts = Counter(token.pos_ for token in doc)
    total = sum(pos_counts.values())

    pos = {
        "noun_ratio": pos_counts["NOUN"] / total if total else 0,
        "verb_ratio": pos_counts["VERB"] / total if total else 0,
        "adj_ratio": pos_counts["ADJ"] / total if total else 0,
        "adv_ratio": pos_counts["ADV"] / total if total else 0
    }

    ner_counts = Counter(ent.label_ for ent in doc.ents)
    ner = {
        "num_ORG": ner_counts["ORG"],
        "num_GPE": ner_counts["GPE"],
        "num_PERSON": ner_counts["PERSON"],
        "num_DATE": ner_counts["DATE"]
    }

    blob = TextBlob(text)
    sentiment = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

    readability = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "avg_sentence_length": textstat.words_per_sentence(text),  # use `sentence_length` to avoid deprecated warning
        "word_count": textstat.lexicon_count(text)
    }

    return {
        "index": article["index"],
        **pos,
        **ner,
        **sentiment,
        **readability,
        "kb_match_score": article.get("kb_match_score", 0),
        "label": article["label"]
    }


def run_feature_extraction():
    with open(input_path, "r", encoding="utf-8") as infile:
        articles = json.load(infile)

    features = [extract_features(a) for a in tqdm(articles, desc="Extracting Features")]

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(features, outfile, indent=2)
