import json
from pathlib import Path
from tqdm import tqdm

# Paths
articles_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_preprocessed_articles.json")
kb_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_knowledge_base.json")
output_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_articles_with_kb_features.json")

# Load data
with articles_path.open("r", encoding="utf-8") as f:
    articles = json.load(f)

with kb_path.open("r", encoding="utf-8") as f:
    kb = json.load(f)

entities = list(kb["entities"].keys())
facts = [fact.lower() for fact in kb["facts"]]  # lowercase for loose match

def match_kb(article_text):
    mentions_known_entity = any(entity.lower() in article_text for entity in entities)
    matches_known_fact = any(fact in article_text for fact in facts)
    return int(mentions_known_entity), int(matches_known_fact)

# Process
kb_articles = []
for article in tqdm(articles):
    text = article["text"]
    ent_flag, fact_flag = match_kb(text)
    article["mentions_known_entity"] = ent_flag
    article["matches_known_fact"] = fact_flag
    kb_articles.append(article)

# Save output
with output_path.open("w", encoding="utf-8") as f:
    json.dump(kb_articles, f, indent=2)

print(f"✅ Saved knowledge-based features to {output_path}")
