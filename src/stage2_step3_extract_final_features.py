import json
import spacy
from textblob import TextBlob
import textstat
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Paths
input_articles_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_preprocessed_articles.json")
knowledge_base_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/knowledge_base.json")
output_features_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json")

# Load models and data
nlp = spacy.load("en_core_web_sm")

with input_articles_path.open("r", encoding="utf-8") as f:
    articles = json.load(f)

with knowledge_base_path.open("r", encoding="utf-8") as f:
    kb = json.load(f)

trusted_entities = set([e.lower() for e in kb["entities"]])
trusted_facts = [fact.lower() for fact in kb["verified_facts"]]

def extract_features(article):
    text = article["text"]
    doc = nlp(text)

    # POS features
    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = sum(pos_counts.values())
    pos_features = {
        "noun_ratio": pos_counts["NOUN"] / total_tokens if total_tokens else 0,
        "verb_ratio": pos_counts["VERB"] / total_tokens if total_tokens else 0,
        "adj_ratio": pos_counts["ADJ"] / total_tokens if total_tokens else 0,
        "adv_ratio": pos_counts["ADV"] / total_tokens if total_tokens else 0,
    }

    # NER features
    ner_counts = Counter([ent.label_ for ent in doc.ents])
    ner_features = {
        "num_ORG": ner_counts["ORG"],
        "num_GPE": ner_counts["GPE"],
        "num_PERSON": ner_counts["PERSON"],
        "num_DATE": ner_counts["DATE"],
    }

    # Sentiment
    blob = TextBlob(text)
    sentiment = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

    # Readability
    readability = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "avg_sentence_length": textstat.words_per_sentence(text),
        "word_count": textstat.lexicon_count(text)
    }

    # Knowledge base matching
    text_lower = text.lower()
    entity_matches = sum(1 for e in trusted_entities if e in text_lower)
    fact_matches = sum(1 for f in trusted_facts if f in text_lower)
    kb_features = {
        "matched_entities": entity_matches,
        "matched_facts": fact_matches
    }

    return {
        "index": article["index"],
        **pos_features,
        **ner_features,
        **sentiment,
        **readability,
        **kb_features,
        "label": article["label"]
    }

# Run extraction
features = [extract_features(article) for article in tqdm(articles)]

with output_features_path.open("w", encoding="utf-8") as f:
    json.dump(features, f, indent=2)

print(f"✅ Final features with KB saved to {output_features_path}")
