from pathlib import Path
from tqdm import tqdm
import spacy
import json
import re

input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/interim/clean_data.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/processed/preprocessed_data.json")

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
stopwords = nlp.Defaults.stop_words


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords])


def preprocess_articles():
    with open(input_path, "r", encoding='utf-8') as infile:
        data = json.load(infile)

    processed_data = []

    for article in tqdm(data, desc="Preprocessing"):
        processed_data.append({
            "index": article["index"],
            "title": preprocess(article["title"]),
            "text": preprocess(article["text"]),
            "label": article["label"],
        })

    with open(output_path, "w", encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, indent=2)