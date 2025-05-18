import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load features
input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json")
with input_path.open("r", encoding="utf-8") as f:
    features = json.load(f)

df = pd.DataFrame(features)

# Encode labels as binary
df["label"] = df["label"].map({"true": 1, "false": 0})

# Split features and labels
X = df.drop(columns=["index", "label"])
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
model_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/src/models/stage2_logreg_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "scaler": scaler, "features": list(X.columns)}, model_path)

print(f"✅ Logistic Regression model saved to {model_path}")