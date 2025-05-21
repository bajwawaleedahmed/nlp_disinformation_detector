from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

original_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/raw/group01_stage1.csv"
predicted_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/results/prediction_results.csv"


def normalize(label):
    return str(label).strip().lower()


def evaluate_predictions():
    original_df = pd.read_csv(original_path)
    predicted_df = pd.read_csv(predicted_path)

    y_true = original_df["real_news"].apply(normalize)
    y_pred = predicted_df["real_news"].apply(normalize)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=2)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print("\n✅ Classification Report:")
    print(report)
    print("✅ Confusion Matrix:")
    print(conf_matrix)
