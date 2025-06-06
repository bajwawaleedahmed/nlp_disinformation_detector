from src.pipeline import data_cleaning, preprocessing, build_kb, extract_features, train_model, predict_model, evaluate_model
from time import sleep


def main():
    print("\n▶ Stage 1: Clean Raw Articles")
    sleep(1)
    data_cleaning.clean_articles()
    sleep(5)

    print("\n▶ Stage 2: Preprocess Articles")
    sleep(1)
    preprocessing.preprocess_articles()
    sleep(5)

    print("\n▶ Stage 3: Build Knowledge Base")
    sleep(1)
    build_kb.build_knowledge_base()
    sleep(5)

    print("\n▶ Stage 4: Extract Features")
    sleep(1)
    extract_features.run_feature_extraction()
    sleep(5)

    print("\n▶ Stage 5: Train Model")
    sleep(1)
    train_model.train_model_xgboost()
    sleep(5)

    print("\n▶ Stage 6: Predict")
    sleep(1)
    predict_model.run_prediction()
    sleep(5)

    print("\n▶ Stage 7: Evaluate")
    evaluate_model.evaluate_predictions()


if __name__ == "__main__":
    main()
