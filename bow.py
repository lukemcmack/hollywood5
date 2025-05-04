import pandas as pd 
import nltk 
from nltk.corpus import stopwords
import re 
import numpy as np 
import heapq 
import os
import unicodedata
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import (
    defaultdict,
)  


#nltk.download("punkt")



def remove_accents(input_str):
    """Remove accents while preserving all other characters"""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', input_str)
        if not unicodedata.combining(c)
    )

def preprocess(text):
	#initiate stopwords
	stop_words_en = set(stopwords.words("english"))
	stop_words_es = set(stopwords.words("spanish"))
	combined_stopwords = stop_words_en.union(stop_words_es)
	stop_words = set(re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in combined_stopwords)
	if isinstance(text, list):
		text = " ".join(text)
	elif not isinstance(text, str):
		return ""

	# Convert to lowercase
	text = text.lower()
	text=remove_accents(text)
    # Remove punctuation
	text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize: split the text into words
	text=re.sub(r"[^a-zA-Z\s]", "",text)
	tokens=nltk.word_tokenize(text)
	return tokens
def rm_stop_words(): 
	

	tokens=[word for word in tokens if word not in stop_words]
	return tokens





def get_vocab(reviews,size): 
	vocab = defaultdict(int)
	for word in reviews:
		vocab[word] += 1
	freq_words = heapq.nlargest(size,vocab, key=vocab.get)
	return freq_words

def popular_words(text,vocab): 
	tokens=[word for word in text if word in vocab]
	return tokens


def vectorize(freq_words,dataset): 
	X = [] 
	for data in dataset: 
		vector = [] 
		for word in freq_words: 
			if word in preprocess(data): 
				vector.append(1) 
			else: 
				vector.append(0) 
		X.append(vector) 
	X = np.asarray(X) 
	return X 









def build_features(df,Year,words): 
	df=df.copy()
	train_df=df[df['Year Nominated'] < Year]
	train_df=train_df.copy()
	train_df.loc[:,'Set']='Train'
	test_df=df[df['Year Nominated'] ==Year]
	test_df=test_df.copy()
	test_df.loc[:,'Set']='Test'
	text=' '.join(train_df['Review Text'].astype(str))
	text=preprocess(text)
	common_words=get_vocab(text,words)
	train_df.loc[:,'clean']=train_df['Review Text'].apply(lambda x: preprocess(x))
	train_df.loc[:,'clean']=train_df['clean'].apply(lambda x: popular_words(x,common_words))
	train_df.loc[:,'clean']=train_df['clean'].apply(lambda x: " ".join(x))
	tfidf_vectorizer = TfidfVectorizer()
	train_tfidf = tfidf_vectorizer.fit_transform(train_df['clean'])
	train_vectors=pd.DataFrame(train_tfidf.toarray(),index=train_df.index ,columns=tfidf_vectorizer.get_feature_names_out())

	test_df.loc[:,'clean']=test_df['Review Text'].apply(lambda x: preprocess(x))
	test_df.loc[:,'clean']=test_df['clean'].apply(lambda x: popular_words(x,common_words))
	test_df.loc[:,'clean']=test_df['clean'].apply(lambda x: " ".join(x))
	test_tfidf = tfidf_vectorizer.fit_transform(test_df['clean'])
	test_vectors=pd.DataFrame(test_tfidf.toarray(),index=test_df.index ,columns=tfidf_vectorizer.get_feature_names_out())

	train_joined = pd.concat([train_df,train_vectors], axis=1).drop(['Review Text','clean'], axis=1)
	test_joined=pd.concat([test_df,test_vectors], axis=1).drop(['Review Text','clean'], axis=1)
	df_combined = pd.concat([train_joined, test_joined], axis=0, ignore_index=True)
	df_combined = df_combined.fillna(0)
	df_combined.to_csv(f"df_{Year}test.csv")
	return df_combined

df_hw = pd.read_csv("data/best_picture_metadata_with_reviews_filtered.csv")























