import pandas as pd
import joblib

# Paths
features_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json"
model_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/src/models/xgboost_model_full.pkl"
output_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_predictions_submission.csv"

# Load data and model (unpack if saved as a tuple)
loaded = joblib.load(model_path)

# If it's a tuple, unpack it
if isinstance(loaded, tuple):
    model = loaded[0]
else:
    model = loaded

# Load feature data
df = pd.read_json(features_path)
X = df.drop(columns=["label", "index"])

# Predict
y_pred = model.predict(X)

# Convert prediction to yes/no
submission = pd.DataFrame({
    "index": df["index"],
    "real_news": ["yes" if p == "true" or p == 1 else "no" for p in y_pred]
})

submission.to_csv(output_path, index=False)
print(f"✅ Submission saved to {output_path}")
