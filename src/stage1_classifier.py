
import json
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

file_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/group01_stage0.json"

# Load dataset
with open(file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Flatten data
print(f"RAW DATA {raw_data}")
df = pd.DataFrame(raw_data)
df_flat = pd.json_normalize(df["articles"])
print(f"FLATTEN DATA {df_flat}")


# Label encoding
df_flat["label"] = df_flat["true"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
X = tfidf.fit_transform(df_flat["text"])
y = df_flat["label"]

# Stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = MultinomialNB()
y_pred = cross_val_predict(model, X, y, cv=skf)

# Evaluation
report = classification_report(y, y_pred)
matrix = confusion_matrix(y, y_pred)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", matrix)
