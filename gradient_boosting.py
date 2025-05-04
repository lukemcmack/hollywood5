import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

RANDOM_STATE = 42
DATA_PATH = r'data/best_picture_metadata_with_reviews_filtered(1).csv'

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Review Text'] = df['Review Text'].astype(str).str.lower()
    df['Year Nominated'] = df['Year Nominated'].astype(int)
    df['Won'] = df['Won'].astype(int)
    return df

def get_custom_stop_words(df):
    stop_words = set(stopwords.words('english'))
    for film in df['Film Name'].unique():
        words = word_tokenize(re.sub(r'[^\w\s]', '', film.lower()))
        stop_words.update(words)
    return list(stop_words)

def evaluate_year(test_year, df, review_vectorizer, clf):
    train_mask = df['Year Nominated'] != test_year
    test_mask = df['Year Nominated'] == test_year

    X_train_text = review_vectorizer.fit_transform(df.loc[train_mask, 'Review Text']).toarray()
    y_train = df.loc[train_mask, 'Won']

    X_test_text = review_vectorizer.transform(df.loc[test_mask, 'Review Text']).toarray()
    y_test = df.loc[test_mask, 'Won']

    clf.fit(X_train_text, y_train)
    y_pred = clf.predict(X_test_text)
    y_prob = clf.predict_proba(X_test_text)[:, 1]

    # Normalize probabilities within the year
    y_prob_normalized = y_prob / y_prob.sum() if y_prob.sum() > 0 else y_prob

    test_df = df[test_mask].copy()
    test_df['win_prob'] = y_prob_normalized

    sorted_df = test_df.sort_values(by='win_prob', ascending=False).reset_index(drop=True)
    predicted_winner = sorted_df.iloc[0]
    actual_winner = sorted_df[sorted_df['Won'] == 1].iloc[0] if any(sorted_df['Won'] == 1) else None

    acc = accuracy_score(y_test, y_pred)
    correct_winner = (actual_winner is not None and predicted_winner['Film Name'] == actual_winner['Film Name'])

    print(f'\n--- Probabilities for {test_year} ---')
    for idx, row in sorted_df.iterrows():
        print(f"{row['Film Name']}: {row['win_prob']:.4f}")

    return {
        'year': test_year,
        'actual_winner': actual_winner['Film Name'] if actual_winner is not None else None,
        'actual_prob': actual_winner['win_prob'] if actual_winner is not None else None,
        'predicted_winner': predicted_winner['Film Name'],
        'accuracy': acc,
        'correct_prediction': correct_winner,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def main():
    df = load_and_preprocess_data(DATA_PATH)

    required_columns = {'Film Name', 'Year Nominated', 'Won', 'Review Text'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    custom_stop_words = get_custom_stop_words(df)

    review_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=custom_stop_words,
        ngram_range=(1, 2)
    )

    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=200)

    results = []

    for test_year in sorted(df['Year Nominated'].unique()):
        print(f"\n=== Evaluating {test_year} ===")
        year_result = evaluate_year(test_year, df, review_vectorizer, clf)
        results.append(year_result)

        print(f"Actual winner: {year_result['actual_winner']}")
        print(f"Predicted winner: {year_result['predicted_winner']}")
        print(f"Correct prediction: {year_result['correct_prediction']}")

    results_df = pd.DataFrame(results)
    avg_accuracy = results_df['accuracy'].mean()
    correct_winners = results_df['correct_prediction'].sum()

    print("\n=== Final Results ===")
    print(f"Correctly predicted winners in {correct_winners} of {len(results_df)} years")
    print(f"Average binary classification accuracy: {avg_accuracy:.3f}")

    # Plot actual winners' predicted probabilities
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['year'], results_df['actual_prob'], marker='o', linestyle='-', color='blue')
    plt.title('Predicted Probability for Actual Winners Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Normalized Predicted Probability')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
