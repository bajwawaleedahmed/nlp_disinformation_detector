import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
original_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/group01_stage1.csv"
predicted_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/stage2_predictions_submission.csv"

# Load both files
original_df = pd.read_csv(original_path)
predicted_df = pd.read_csv(predicted_path)

# Normalize labels
def normalize(label):
    return str(label).strip().lower()

y_true = original_df["real_news"].apply(normalize)
y_pred = predicted_df["real_news"].apply(normalize)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=2)
conf_matrix = confusion_matrix(y_true, y_pred)

# Print results
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print("\n✅ Classification Report:")
print(report)
print("✅ Confusion Matrix:")
print(conf_matrix)
