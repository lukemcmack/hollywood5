import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/bp_embedding_data.csv")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
#model = SentenceTransformer("all-MiniLM-L12-v2")


# Precompute all review-level embeddings
print("Encoding all reviews...")
df["embedding"] = model.encode(df["Review Text"].fillna("").tolist(), show_progress_bar=True).tolist()

def evaluate_year(test_year, df):
    """Evaluate predictions for a single test year using sentence embeddings"""
    train_df = df[df["Year Nominated"] != test_year].copy()
    test_df = df[df["Year Nominated"] == test_year].copy()

    # Aggregate review embeddings to movie level
    train_movie_embeddings = train_df.groupby("Film Name")["embedding"].apply(
        lambda embs: np.mean(np.vstack(embs), axis=0)
    ).reset_index()

    test_movie_embeddings = test_df.groupby("Film Name")["embedding"].apply(
        lambda embs: np.mean(np.vstack(embs), axis=0)
    ).reset_index()

    # Merge with labels
    train_labels = train_df[["Film Name", "Won"]].drop_duplicates()
    test_labels = test_df[["Film Name", "Won"]].drop_duplicates()

    train_data = pd.merge(train_movie_embeddings, train_labels, on="Film Name")
    test_data = pd.merge(test_movie_embeddings, test_labels, on="Film Name")

    X_train = np.vstack(train_data["embedding"])
    y_train = train_data["Won"]

    X_test = np.vstack(test_data["embedding"])
    y_test = test_data["Won"]

    # Fit model
    clf = LogisticRegressionCV(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Predicted and actual winner logic
    test_data["Predicted Probability"] = y_proba
    predicted_winner = test_data.loc[test_data["Predicted Probability"].idxmax()]
    actual_winner = test_data[test_data["Won"] == 1].iloc[0] if any(test_data["Won"] == 1) else None

    correct_prediction = (
        actual_winner is not None and predicted_winner["Film Name"] == actual_winner["Film Name"]
    )

    acc = accuracy_score(y_test, y_pred)

    return {
        "year": test_year,
        "actual_winner": actual_winner["Film Name"] if actual_winner is not None else None,
        "predicted_winner": predicted_winner["Film Name"],
        "correct_prediction": correct_prediction,
        "accuracy": acc,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "test_data": test_data,
        "clf": clf,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }

# Run evaluation for all years
years = sorted(df["Year Nominated"].unique())
results = []

for test_year in years:
    print(f"\n=== Evaluating {test_year} ===")
    year_result = evaluate_year(test_year, df)
    results.append(year_result)

    print(f"Actual winner: {year_result['actual_winner']}")
    print(f"Predicted winner: {year_result['predicted_winner']}")
    print(f"Correct prediction: {year_result['correct_prediction']}")

# Summary results
results_df = pd.DataFrame(results)[["year", "actual_winner", "predicted_winner", "correct_prediction", "accuracy"]]
avg_accuracy = results_df["accuracy"].mean()
total_correct = results_df["correct_prediction"].sum()

print("\n=== Final Summary ===")
print(f"Correctly predicted winners in {total_correct} of {len(years)} years")
print(f"Average accuracy: {avg_accuracy:.3f}")

plt.figure(figsize=(12, 6))
plt.plot(results_df['year'], results_df['actual_prob'], marker='o', linestyle='-', color='blue', label='Actual Winner')
plt.plot(results_df['year'], results_df['predicted_prob'], marker='s', linestyle='--', color='red', label='Predicted Winner')
plt.title('Gradient Boosting Probabilities: Actual vs Predicted Winners Over the Years')
plt.xlabel('Year')
plt.ylabel('Normalized Predicted Probability')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()
