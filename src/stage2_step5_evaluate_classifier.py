import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
features_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json")
model_path = Path("models/stage2_best_model.pkl")

# Load data and model
with features_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["label"] = df["label"].map({"true": 1, "false": 0})

X = df.drop(columns=["index", "label"])
y = df["label"]

# Split (same seed as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load model
model = joblib.load(model_path)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=["false", "true"]))

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
