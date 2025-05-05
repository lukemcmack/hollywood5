import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob

# Configuration

DATA_PATH = r'data\best_picture_metadata_with_sampled_english_reviews.csv'

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Review Text'] = df['Review Text'].astype(str).str.lower()
    df['Year Nominated'] = df['Year Nominated'].astype(int)
    df['Won'] = df['Won'].astype(int)
    
    # Enhanced text cleaning
    df['Review Text'] = df['Review Text'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', re.sub(r'\d+', '', x))
    )
    
    # Add sentiment analysis features
    df = add_sentiment(df)
    return df

def add_sentiment(df):
    df['sentiment'] = df['Review Text'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df['subjectivity'] = df['Review Text'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )
    return df

def get_custom_stop_words(df):
    stop_words = set(stopwords.words('english'))
    
    # Remove common review fluff words
    stop_words.update(['film', 'movie', 'story', 'character', 'director', 
                      'performance', 'scene', 'films', 'movies', 'watch',
                      'oscar', 'award', 'best', 'picture'])
    
    # Keep film names but remove common words from them
    for film in df['Film Name'].unique():
        words = [w for w in word_tokenize(re.sub(r'[^\w\s]', '', film.lower())) 
                if len(w) > 2 and w not in ['man', 'woman', 'day', 'night', 'love', 'time']]
        stop_words.update(words)
    
    return list(stop_words)

def temporal_weight(train_year, predict_year):
    # More gradual decay to capture longer trends
    return np.exp(-abs(train_year - predict_year)/8) + 0.2  # Baseline importance

def evaluate_year(test_year, df, review_vectorizer):
    train_mask = df['Year Nominated'] != test_year
    test_mask = df['Year Nominated'] == test_year
    
    if sum(train_mask) == 0 or sum(test_mask) == 0:
        return None
    
    # Calculate temporal weights
    train_years = df.loc[train_mask, 'Year Nominated']
    weights = temporal_weight(train_years, test_year)
    
    # Feature engineering
    X_train_text = review_vectorizer.fit_transform(df.loc[train_mask, 'Review Text'])
    y_train = df.loc[train_mask, 'Won']
    
    X_test_text = review_vectorizer.transform(df.loc[test_mask, 'Review Text'])
    
    # Temporal-Weighted Naive Bayes with adjusted alpha
    clf = MultinomialNB(alpha=0.75)
    clf.fit(X_train_text, y_train, sample_weight=weights)
    
    # Get probabilities
    y_prob = clf.predict_proba(X_test_text)[:, 1]
    y_prob_normalized = y_prob / (y_prob.sum() + 1e-6)

    test_df = df[test_mask].copy()
    test_df['win_prob'] = y_prob_normalized
    
    # Apply sentiment adjustments
    positive_mask = (test_df['sentiment'] > 0.3) & (test_df['subjectivity'] > 0.5)
    test_df.loc[positive_mask, 'win_prob'] *= 1.2
    
    length_mask = test_df['Review Text'].str.split().str.len() < 50
    test_df.loc[length_mask, 'win_prob'] *= 0.8
    
    # Final normalization
    test_df['win_prob'] = test_df['win_prob'] / test_df['win_prob'].sum()
    
    sorted_df = test_df.sort_values('win_prob', ascending=False)
    predicted_winner = sorted_df.iloc[0]
    actual_winners = test_df[test_df['Won'] == 1]
    
    return {
        'year': test_year,
        'actual_winner': actual_winners.iloc[0]['Film Name'] if len(actual_winners) > 0 else None,
        'actual_prob': actual_winners.iloc[0]['win_prob'] if len(actual_winners) > 0 else None,
        'predicted_winner': predicted_winner['Film Name'],
        'predicted_prob': predicted_winner['win_prob'],
        'correct': predicted_winner['Film Name'] in actual_winners['Film Name'].values,
        'model': clf  # Return model for feature inspection
    }

def weighted_naive_bayes(DATA_PATH):
    RANDOM_STATE = 42
    df = load_and_preprocess_data(DATA_PATH)
    
    # Verify data
    if df.groupby('Year Nominated')['Won'].sum().min() == 0:
        print("Warning: Some years have no winners marked!")
    
    custom_stop_words = get_custom_stop_words(df)
    
    # Optimized vectorizer
    review_vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words=custom_stop_words,
        ngram_range=(1, 3),  # Include trigrams
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
        analyzer='word',
        token_pattern=r'(?u)\b[\w-]+\b'  # Keep hyphenated words
    )
    
    results = []
    models = {}  # Store models by year
    
    for test_year in sorted(df['Year Nominated'].unique()):
        print(f"\n=== Evaluating {test_year} ===")
        year_result = evaluate_year(test_year, df, review_vectorizer)
        if year_result:
            results.append(year_result)
            models[test_year] = year_result['model']
            print(f"Predicted: {year_result['predicted_winner']} ({year_result['predicted_prob']:.2f})")
            print(f"Actual: {year_result['actual_winner']} ({year_result.get('actual_prob', 'N/A')})")
            print(f"Correct: {year_result['correct']}")

    results_df = pd.DataFrame(results)
    
    # Enhanced visualization
  # [Previous code remains exactly the same until the visualization section]

    # Simplified two-line visualization
    plt.figure(figsize=(12, 6))
    
    # Actual winners (blue solid line)
    plt.plot(results_df['year'], results_df['actual_prob'], 
             marker='o', linestyle='-', color='blue',
             label='Actual Winner', linewidth=2)
    
    # Predicted winners (red dashed line) 
    plt.plot(results_df['year'], results_df['predicted_prob'],
             marker='s', linestyle='--', color='red',
             label='Predicted Winner', linewidth=2)
    
    plt.title('Oscar Winner Prediction Probabilities (2015-2025)')
    plt.xlabel('Year')
    plt.ylabel('Normalized Probability')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temporal_weighted_naive_bayes_predictions.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Print results and features
    if len(results) > 0:
        accuracy = results_df['correct'].mean()
        print(f"\nFinal Accuracy: {accuracy:.1%}")
        
        # Show top predictive features across all years
        print("\nTop Predictive Features Across All Years:")
        all_features = {}
        for year, model in models.items():
            feature_names = review_vectorizer.get_feature_names_out()
            log_probs = model.feature_log_prob_[1]
            for feat, score in zip(feature_names, log_probs):
                all_features[feat] = all_features.get(feat, 0) + np.exp(score)
        
        top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:30]
        for feat, score in top_features:
            print(f"{feat}: {score/len(models):.4f} (avg score)")




