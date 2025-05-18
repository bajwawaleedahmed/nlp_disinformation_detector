import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Paths
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json")
model_path = Path("models/stage2_best_model.pkl")

# Load features
with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Encode labels (true → 1, false → 0)
df["label"] = df["label"].map({"true": 1, "false": 0})

# Drop index and keep only feature columns
X = df.drop(columns=["index", "label"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Trained Random Forest model saved to {model_path}")
