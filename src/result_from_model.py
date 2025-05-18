import pandas as pd
import joblib

# Paths
features_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_features_final.json"
model_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/src/models/stage2_best_model.pkl"
output_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_predictions_submission.csv"

# Load data and model
df = pd.read_json(features_path)
model = joblib.load(model_path)

# Prepare feature matrix
X = df.drop(columns=["label", "index"])
y_pred = model.predict(X)

# Format and save submission
submission = pd.DataFrame({
    "index": df["index"],
    "real_news": ["yes" if p == "true" or p == 1 else "no" for p in y_pred]
})
submission.to_csv(output_path, index=False)

print(f"✅ Submission saved to {output_path}")