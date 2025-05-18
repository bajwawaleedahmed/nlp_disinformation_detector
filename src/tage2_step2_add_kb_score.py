import json
import spacy
from tqdm import tqdm
from pathlib import Path

# Use lightweight pipeline
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

# Define climate-related knowledge base
CLIMATE_FACTS = {
    "global temperature rise": ["1.5°c", "2°c", "warming", "temperature", "global heating"],
    "greenhouse gases": ["carbon dioxide", "co2", "methane", "greenhouse", "emissions"],
    "sea level rise": ["sea", "ice", "melting", "glacier", "flooding"],
    "climate organizations": ["ipcc", "nasa", "unep", "noaa", "wmo"],
    "climate treaties": ["paris", "cop26", "kyoto"]
}

def compute_factual_match_score(text):
    doc = nlp(text.lower())
    tokens = set(token.text for token in doc if token.is_alpha)
    total_hits = sum(any(kw in tokens for kw in keywords) for keywords in CLIMATE_FACTS.values())
    return total_hits / len(CLIMATE_FACTS)

# File paths
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_preprocessed_articles.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_with_kbscore.json")

# Process and annotate
with input_path.open("r", encoding="utf-8") as infile:
    articles = json.load(infile)

for article in tqdm(articles):
    score = compute_factual_match_score(article["text"])
    article["kb_match_score"] = score

with output_path.open("w", encoding="utf-8") as outfile:
    json.dump(articles, outfile, indent=2)

print(f"✅ KB score added and saved to: {output_path}")
