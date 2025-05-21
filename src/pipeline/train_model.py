from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import xgboost as xgb
import pandas as pd
import joblib
import json

input_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/data/processed/preprocessed_data_with_features.json")
model_path = Path("/home/chief/PycharmProjects/nlp_disinformation_detector/models/trained_model.pkl")


def train_model_random_forest():
    with input_path.open("r", encoding="utf-8") as f:
        features = json.load(f)

    df = pd.DataFrame(features)

    df["label"] = df["label"].map({"true": 1, "false": 0})

    feature_cols = [col for col in df.columns if col not in ["index", "label"]]

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print("Random Forest Training Complete")


def train_model_xgboost():
    df = pd.read_json(input_path)

    X = df.drop(columns=["label", "index"])
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X, y_encoded)
    joblib.dump((model, le), model_path)
    print("XGBoost Training Complete")
