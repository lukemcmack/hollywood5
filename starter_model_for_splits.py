import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("all")

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "best_picture_metadata_with_reviews_filtered.csv")


def load_and_preprocess_data(filepath):
    """Load and preprocess the Oscar nomination data"""
    df = pd.read_csv(filepath)

    text_columns = ["Description", "Review Text"]
    for col in text_columns:
        df[col] = df[col].astype(str).str.lower()

    df["Year Nominated"] = df["Year Nominated"].astype(int)
    df["Won"] = df["Won"].astype(int)

    return df


def get_custom_stop_words(df):
    """Create comprehensive stop words list including film names, cast, studios"""
    stop_words = set(stopwords.words("english"))

    for film in df["Film Name"].unique():
        words = word_tokenize(re.sub(r"[^\w\s]", "", film.lower()))
        stop_words.update(words)

    for cast_str in df["Cast"].dropna():
        for actor in cast_str.split(","):
            words = word_tokenize(re.sub(r"[^\w\s]", "", actor.strip().lower()))
            stop_words.update(words)

    for studio in df["Studios"].dropna().unique():
        words = word_tokenize(re.sub(r"[^\w\s]", "", studio.lower()))
        stop_words.update(words)

    return list(stop_words)


def evaluate_year(test_year, df, model):
    """Evaluate performance for a specific test year"""
    train_mask = df["Year Nominated"] != test_year
    test_mask = df["Year Nominated"] == test_year

    X_train = df[train_mask]
    y_train = df[train_mask]["Won"]
    X_test = df[test_mask]
    y_test = df[test_mask]["Won"]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_df = df[test_mask].copy()
    test_df["win_prob"] = y_prob
    predicted_winner = test_df.loc[test_df["win_prob"].idxmax()]
    actual_winner = (
        test_df[test_df["Won"] == 1].iloc[0] if any(test_df["Won"] == 1) else None
    )

    acc = accuracy_score(y_test, y_pred)
    correct_winner = (
        actual_winner is not None and predicted_winner.name == actual_winner.name
    )

    return {
        "year": test_year,
        "actual_winner": (
            actual_winner["Film Name"] if actual_winner is not None else None
        ),
        "predicted_winner": predicted_winner["Film Name"],
        "accuracy": acc,
        "correct_prediction": correct_winner,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }


def main():
    df = load_and_preprocess_data(DATA_PATH)

    required_columns = {"Film Name", "Year Nominated", "Won", "Review Text"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    custom_stop_words = get_custom_stop_words(df)

    text_preprocessor = Pipeline(
        [
            ("clean", FunctionTransformer),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000, stop_words=custom_stop_words, ngram_range=(1, 2)
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("text_reviews", text_preprocessor, "Review Text"),
            ("text_description", text_preprocessor, "Description"),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000
                ),
            ),
        ]
    )

    years = sorted(df["Year Nominated"].unique())
    results = []

    for test_year in years:
        print(f"\n=== Evaluating {test_year} ===")
        year_result = evaluate_year(test_year, df, model)
        results.append(year_result)

        print(f"Actual winner: {year_result['actual_winner']}")
        print(f"Predicted winner: {year_result['predicted_winner']}")
        print(f"Correct prediction: {year_result['correct_prediction']}")

    results_df = pd.DataFrame(results)

    avg_accuracy = results_df["accuracy"].mean()
    correct_winners = results_df["correct_prediction"].sum()

    print("\n=== Final Results ===")
    print(f"Correctly predicted winners in {correct_winners} of {len(years)} years")


if __name__ == "__main__":
    main()
