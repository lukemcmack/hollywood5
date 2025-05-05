import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("data/bp_embedding_data_100.csv")
model = SentenceTransformer("all-MiniLM-L12-v2")

print("Encoding all reviews...")
df["embedding"] = model.encode(df["Review Text"].fillna("").tolist(), show_progress_bar=True).tolist()

results = []
all_predictions = []
years = sorted(df["Year Nominated"].unique())

for test_year in years:
    print(f"\n=== Evaluating {test_year} ===")

    train_df = df[df["Year Nominated"] != test_year].copy()
    test_df = df[df["Year Nominated"] == test_year].copy()

    train_embeddings = train_df.groupby("Film Name")["embedding"].apply(
        lambda embs: np.mean(np.vstack(embs), axis=0)
    ).reset_index()

    test_embeddings = test_df.groupby("Film Name")["embedding"].apply(
        lambda embs: np.mean(np.vstack(embs), axis=0)
    ).reset_index()

    train_labels = train_df[["Film Name", "Won"]].drop_duplicates()
    test_labels = test_df[["Film Name", "Won"]].drop_duplicates()

    train_data = pd.merge(train_embeddings, train_labels, on="Film Name")
    test_data = pd.merge(test_embeddings, test_labels, on="Film Name")

    X_train = np.vstack(train_data["embedding"])
    y_train = train_data["Won"]
    X_test = np.vstack(test_data["embedding"])
    y_test = test_data["Won"]

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    test_data["Predicted Probability"] = y_proba
    test_data["Year"] = test_year

    all_predictions.append(test_data[["Year", "Film Name", "Won", "Predicted Probability"]])

    predicted_winner = test_data.loc[test_data["Predicted Probability"].idxmax()]
    actual_winner = test_data[test_data["Won"] == 1].iloc[0] if any(test_data["Won"] == 1) else None

    correct = actual_winner is not None and predicted_winner["Film Name"] == actual_winner["Film Name"]
    actual_prob = (
        test_data.loc[test_data["Film Name"] == actual_winner["Film Name"], "Predicted Probability"].values[0]
        if actual_winner is not None else None
    )

    results.append({
        "year": test_year,
        "actual_winner": actual_winner["Film Name"] if actual_winner is not None else None,
        "predicted_winner": predicted_winner["Film Name"],
        "correct_prediction": correct,
        "accuracy": accuracy_score(y_test, clf.predict(X_test)),
        "actual_prob": actual_prob,
        "predicted_prob": predicted_winner["Predicted Probability"]
    })

    print(f"Actual winner: {actual_winner['Film Name'] if actual_winner is not None else 'None'}")
    print(f"Predicted winner: {predicted_winner['Film Name']}")
    print(f"Correct prediction: {correct}")

results_df = pd.DataFrame(results)
avg_accuracy = results_df["accuracy"].mean()
total_correct = results_df["correct_prediction"].sum()

print("\n=== Final Summary ===")
print(f"Correctly predicted winners in {total_correct} of {len(years)} years")
print(f"Average accuracy: {avg_accuracy:.3f}")

all_predictions_df = pd.concat(all_predictions, ignore_index=True)
all_predictions_df = all_predictions_df.sort_values(by=["Year", "Predicted Probability"], ascending=[True, False])

print("\n=== Top 3 Predicted Movies Per Year ===")
print(all_predictions_df.groupby("Year").head(3).to_string(index=False))

plt.figure(figsize=(12, 6))
plt.plot(results_df["year"], results_df["actual_prob"], marker="o", linestyle="-", label="Actual Winner Probability")
plt.plot(results_df["year"], results_df["predicted_prob"], marker="s", linestyle="--", label="Predicted Winner Probability")
plt.title("Predicted Probabilities: Actual vs Predicted Winners")
plt.xlabel("Year")
plt.ylabel("Predicted Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally save predictions
# all_predictions_df.to_csv("all_movie_predictions.csv", index=False)
# results_df.to_csv("summary_predictions.csv", index=False)
