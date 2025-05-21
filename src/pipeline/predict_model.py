import pandas as pd
import joblib

features_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/processed/preprocessed_data_with_features.json"
model_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/models/trained_model.pkl"
output_path = "/home/chief/PycharmProjects/nlp_disinformation_detector/data/results/prediction_results.csv"


def run_prediction():
    loaded = joblib.load(model_path)

    if isinstance(loaded, tuple):
        model = loaded[0]
    else:
        model = loaded

    df = pd.read_json(features_path)
    X = df.drop(columns=["label", "index"])

    y_pred = model.predict(X)

    submission = pd.DataFrame({
        "index": df["index"],
        "real_news": ["yes" if p == "true" or p == 1 else "no" for p in y_pred]
    })

    submission.to_csv(output_path, index=False)
    print("Prediction results generated")
