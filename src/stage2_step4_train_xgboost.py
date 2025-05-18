import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

# Paths
features_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json"
model_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/src/models/xgboost_model_full.pkl"

# Load dataset
df = pd.read_json(features_path)

# Prepare features and labels
X = df.drop(columns=["label", "index"])
y = df["label"]

# Encode labels to 0/1
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'false' -> 0, 'true' -> 1

# Train XGBoost model on full data
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X, y_encoded)

# Save model and label encoder
joblib.dump((model, le), model_path)
print(f"✅ Full model trained and saved to {model_path}")
