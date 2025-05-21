import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stopwords = nlp.Defaults.stop_words


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords])
