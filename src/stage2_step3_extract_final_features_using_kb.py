import json
import spacy
from textblob import TextBlob
import textstat
from tqdm import tqdm
from pathlib import Path
from collections import Counter

# Load full spaCy pipeline
nlp = spacy.load("en_core_web_sm")

# File paths
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_with_kbscore.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json")

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

# Load and extract
with input_path.open("r", encoding="utf-8") as infile:
    articles = json.load(infile)

features = [extract_features(a) for a in tqdm(articles)]

with output_path.open("w", encoding="utf-8") as outfile:
    json.dump(features, outfile, indent=2)

print(f"✅ Final features with KB saved to {output_path}")
