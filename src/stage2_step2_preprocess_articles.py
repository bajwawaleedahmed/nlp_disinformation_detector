import json
import spacy
import re
from tqdm import tqdm
from pathlib import Path

# Load spaCy model without parser and NER for speed
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stopwords = nlp.Defaults.stop_words

def preprocess(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords])

# Define paths using pathlib
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_clean_articles.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_preprocessed_articles.json")

# Load, process, and save data
with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
    articles = json.load(infile)
    outfile.write("[\n")

    for i, article in enumerate(tqdm(articles, desc="Preprocessing")):
        cleaned = {
            "index": article["index"],
            "title": preprocess(article["title"]),
            "text": preprocess(article["text"]),
            "label": article["label"]
        }
        json.dump(cleaned, outfile, ensure_ascii=False)
        if i < len(articles) - 1:
            outfile.write(",\n")
        else:
            outfile.write("\n")

    outfile.write("]\n")

print(f"✅ Saved preprocessed articles to {output_path}")
