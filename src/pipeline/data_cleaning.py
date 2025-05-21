from pathlib import Path
from tqdm import tqdm
import json

input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/raw/stage2_clean_articles.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/interim/clean_data.json")


def clean_articles():
    with open(input_path, "r", encoding="utf-8") as infile:
        raw_articles = json.load(infile)

    cleaned_articles = []

    for article in tqdm(raw_articles, desc="Cleaning articles"):
        cleaned_article = {
            "index": article["index"],
            "title": article["title"],
            "text": article["text"],
            "label": article["label"],
        }
        cleaned_articles.append(cleaned_article)

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(cleaned_articles, outfile, indent=2)