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
nltk.download('all')

# Configuration
RANDOM_STATE = 42
DATA_PATH = r'data\best_picture_metadata_with_reviews_filtered.csv'  # Update with your actual file path

def load_and_preprocess_data(filepath):
    """Load and preprocess the Oscar nomination data"""
    df = pd.read_csv(filepath)
    
    # Clean text data
    text_columns = ['Description', 'Review Text']
    for col in text_columns:
        df[col] = df[col].astype(str).str.lower()
    
    # Convert relevant columns
    df['Year Nominated'] = df['Year Nominated'].astype(int)
    df['Won'] = df['Won'].astype(int)
    
    return df

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

def evaluate_year(test_year, df, model):
    """Evaluate performance for a specific test year"""
    train_mask = df['Year Nominated'] != test_year
    test_mask = df['Year Nominated'] == test_year
    
    X_train = df[train_mask]
    y_train = df[train_mask]['Won']
    X_test = df[test_mask]
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

def main():
    # Load data
    df = load_and_preprocess_data(DATA_PATH)
    
    # Verify required columns
    required_columns = {'Film Name', 'Year Nominated', 'Won', 'Review Text'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Get custom stop words
    custom_stop_words = get_custom_stop_words(df)
    
    # Create text preprocessing pipeline
    text_preprocessor = Pipeline([
        ('clean', FunctionTransformer),
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            stop_words=custom_stop_words,
            ngram_range=(1, 2)))
    ])
    
    # Create column transformer for multiple features
    preprocessor = ColumnTransformer([
        ('text_reviews', text_preprocessor, 'Review Text'),
        ('text_description', text_preprocessor, 'Description'),
    ])
    
    # Full model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=1000))
    ])
    
    # Evaluate for each year
    years = sorted(df['Year Nominated'].unique())
    results = []
    
    for test_year in years:
        print(f"\n=== Evaluating {test_year} ===")
        year_result = evaluate_year(test_year, df, model)
        results.append(year_result)
        
        print(f"Actual winner: {year_result['actual_winner']}")
        print(f"Predicted winner: {year_result['predicted_winner']}")
        print(f"Correct prediction: {year_result['correct_prediction']}")
        print(f"Accuracy: {year_result['accuracy']:.2f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate and print overall metrics
    avg_accuracy = results_df['accuracy'].mean()
    correct_winners = results_df['correct_prediction'].sum()
    
    print("\n=== Final Results ===")
    print(f"Average accuracy across years: {avg_accuracy:.2f}")
    print(f"Correctly predicted winners in {correct_winners} of {len(years)} years")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['year'].astype(str), results_df['accuracy'])
    plt.axhline(avg_accuracy, color='red', linestyle='--', label=f'Average: {avg_accuracy:.2f}')
    
    # Add correct prediction markers
    for i, correct in enumerate(results_df['correct_prediction']):
        if correct:
            bars[i].set_color('green')

if __name__ == "__main__":
    main()