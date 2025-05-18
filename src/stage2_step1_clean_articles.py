# Re-run after kernel reset

import json
import spacy
import string

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define stopwords and punctuation
stopwords = nlp.Defaults.stop_words
punctuation = string.punctuation

# Load the cleaned article data
input_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_clean_articles.json"
output_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_clean_articles_preprocessed.json"

with open(input_path, "r", encoding="utf-8") as f:
    articles = json.load(f)


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in stopwords
           and token.lemma_ not in punctuation
           and token.is_alpha
    ]
    return " ".join(tokens)


# Apply preprocessing to each article
for article in articles:
    article["clean_title"] = preprocess_text(article["title"])
    article["clean_text"] = preprocess_text(article["text"])

# Save the preprocessed data
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2)

output_path  # Return the path to the preprocessed file
