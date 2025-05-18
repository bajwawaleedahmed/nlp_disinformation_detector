import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json")
model_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/src/models/stage2_final_model.pkl")

# Load features
with input_path.open("r", encoding="utf-8") as f:
    features = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(features)

# Encode labels (true/false → 1/0)
df["label"] = df["label"].map({"true": 1, "false": 0})

# Define feature columns
feature_cols = [col for col in df.columns if col not in ["index", "label"]]

# Prepare data
X = df[feature_cols]
y = df["label"]

# Split (optional, here for test sanity)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Final model trained and saved to {model_path}")
