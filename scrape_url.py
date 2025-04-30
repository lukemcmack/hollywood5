import os
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import markdown
import json
import numpy as np
import pandas as pd
from openai import AuthenticationError

nominees=pd.read_csv("OscarWinners.csv")

unique_movies=nominees[["Movie Title","Year"]].drop_duplicates()

'''for _, i in unique_movies.iterrows(): 
	print(f"{i['Movie Title']} {i['Year']} ")'''


def run_Perplexity(CONTENT, PROMPT):
	load_dotenv()
	PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
	client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
	messages = [
		{
			"role": "system",
			"content": CONTENT,
		},
		{
			"role": "user",
			"content": PROMPT,
		},
	]

	completion = client.chat.completions.create(
	model="sonar", messages=messages)
	return completion.choices[0].message.content






def get_url(movie_df): 
    unique_movies = movie_df.copy()
    unique_movies.loc[:, 'url'] = None

    for idx, row in unique_movies.iterrows(): 
        prompt = f"{row['Movie Title']} {row['Year']}"
        try:
	        url = run_Perplexity(
	            "Provide only the full url for the Letterboxd website for the following movie. No other info.",
	            prompt
	        )

        	unique_movies.loc[idx, 'url'] = url
        except AuthenticationError as e:
        	print(f"Authentication failed: {e}")
        	break

    return unique_movies


with_url=get_url(unique_movies.head())

with_url.to_excel("exceptURL.xlsx",index=False)













