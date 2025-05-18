import json
import re
from pathlib import Path
from tqdm import tqdm
import spacy

# ✅ Load spaCy with only the tagger and tokenizer
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stopwords = nlp.Defaults.stop_words

# ✅ File paths
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_clean_articles.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_preprocessed_articles.json")

# ✅ Text cleaning and lemmatization
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    doc = nlp(text)
    return " ".join([
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_ not in stopwords
    ])

# ✅ Load, preprocess, and write to new file
with input_path.open("r", encoding="utf-8") as infile:
    articles = json.load(infile)

preprocessed = []

for article in tqdm(articles, desc="Preprocessing"):
    processed = {
        "index": article["index"],
        "title": preprocess(article["title"]),
        "text": preprocess(article["text"]),
        "label": article["label"].lower()  # Ensuring label is 'true'/'false'
    }
    preprocessed.append(processed)

# ✅ Save output
with output_path.open("w", encoding="utf-8") as outfile:
    json.dump(preprocessed, outfile, indent=2)

print(f"✅ Saved preprocessed articles to {output_path}")
