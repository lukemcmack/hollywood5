import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")


def evaluate_year(test_year, df, model):
    """Evaluate performance for a specific test year"""
    train_mask = df['Year Nominated'] != test_year
    test_mask = df['Year Nominated'] == test_year
    
    X_train = df[train_mask]["Review Text"].fillna("")
    y_train = df[train_mask]['Won']
    X_test = df[test_mask]["Review Text"].fillna("")
    y_test = df[test_mask]['Won']
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Get predicted and actual winners
    test_df = df[test_mask].copy()
    test_df['win_prob'] = y_prob
    predicted_winner = test_df.loc[test_df['win_prob'].idxmax()]
    actual_winner = test_df[test_df['Won'] == 1].iloc[0] if any(test_df['Won'] == 1) else None
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    correct_winner = (actual_winner is not None and 
                     predicted_winner.name == actual_winner.name)
    
    return {
        'year': test_year,
        'actual_winner': actual_winner['Film Name'] if actual_winner is not None else None,
        'predicted_winner': predicted_winner['Film Name'],
        'accuracy': acc,
        'correct_prediction': correct_winner,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def get_custom_stop_words(df):
    """Create comprehensive stop words list including film names, cast, studios"""
    # Basic English stop words
    stop_words = set(stopwords.words('english'))
    
    # Add film names
    for film in df['Film Name'].unique():
        words = word_tokenize(re.sub(r'[^\w\s]', '', film.lower()))
        stop_words.update(words)
    
    # Add cast members
    for cast_str in df['Cast'].dropna():
        for actor in cast_str.split(','):
            words = word_tokenize(re.sub(r'[^\w\s]', '', actor.strip().lower()))
            stop_words.update(words)
    
    # Add studios
    for studio in df['Studios'].dropna().unique():
        words = word_tokenize(re.sub(r'[^\w\s]', '', studio.lower()))
        stop_words.update(words)
    
    return list(stop_words)

df = pd.read_csv("data/best_picture_metadata_with_sampled_english_reviews.csv")

years = sorted(df['Year Nominated'].unique())
results = []
custom_stop_words = get_custom_stop_words(df)

model = make_pipeline(
    CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=custom_stop_words,
        ngram_range=(1, 2),  # This sets unigrams and bigrams
        min_df= 10
    ),
    LogisticRegressionCV(max_iter=1000)  # Uses ridge (L2) regularization by default
)



for test_year in years:
        print(f"\n=== Evaluating {test_year} ===")
        year_result = evaluate_year(test_year, df, model)
        results.append(year_result)
        
        print(f"Actual winner: {year_result['actual_winner']}")
        print(f"Predicted winner: {year_result['predicted_winner']}")
        print(f"Correct prediction: {year_result['correct_prediction']}")
    
results_df = pd.DataFrame(results)
    
avg_accuracy = results_df['accuracy'].mean()
correct_winners = results_df['correct_prediction'].sum()
    
print("\n=== Final Results ===")
print(f"Correctly predicted winners in {correct_winners} of {len(years)} years")

y_test_pred = model.predict(X_test)

RocCurveDisplay.from_estimator(model, X_test, y=y_test)
PrecisionRecallDisplay.from_estimator(model, X_test, y=y_test)

print(classification_report(y_true=y_test, y_pred=y_test_pred))
print()

cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

test_df = test_df.copy() 
test_df["Predicted Won"] = y_test_pred

y_test_proba = model.predict_proba(X_test)[:, 1]


test_df = test_df.copy()
test_df["Predicted Won"] = y_test_pred
test_df["Predicted Probability"] = y_test_proba

winner_idx = test_df["Predicted Probability"].idxmax()

test_df["Predicted Won (Max Only)"] = 0

test_df.loc[winner_idx, "Predicted Won (Max Only)"] = 1

print(test_df[["Film Name", "Won", "Predicted Won (Max Only)", "Predicted Probability"]])

plt.figure(figsize=(10, 6))
plt.plot(results_df['year'], results_df['actual_prob'], marker='o', linestyle='-', color='blue')
plt.title('Predicted Probability for Actual Winners Over the Years')
plt.xlabel('Year')
plt.ylabel('Normalized Predicted Probability')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
