from pathlib import Path
from tqdm import tqdm
import spacy
import json

input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/processed/preprocessed_data.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/processed/preprocessed_data_with_kb.json")

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

CLIMATE_FACTS = {
    "global temperature rise": ["1.5°c", "2°c", "warming", "temperature", "global heating"],
    "greenhouse gases": ["carbon dioxide", "co2", "methane", "greenhouse", "emissions"],
    "sea level rise": ["sea", "ice", "melting", "glacier", "flooding"],
    "climate organizations": ["ipcc", "nasa", "unep", "noaa", "wmo"],
    "climate treaties": ["paris", "cop26", "kyoto"]
}


def compute_kb_score(text):
    doc = nlp(text.lower())
    tokens = set(token.text for token in doc if token.is_alpha)
    total_hits = sum(any(kw in tokens for kw in keywords) for keywords in CLIMATE_FACTS.values())
    return total_hits / len(CLIMATE_FACTS)


def build_knowledge_base():
    with open(input_path, "r", encoding="utf-8") as infile:
        articles = json.load(infile)

    for article in tqdm(articles, desc="Building KB"):
        score = compute_kb_score(article["text"])
        article["kb_match_score"] = score

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(articles, outfile, indent=2)