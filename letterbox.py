import requests
from bs4 import BeautifulSoup
from ftfy import fix_text
import pandas as pd 
import re
def get_soup(url):

    """Takes a URL and returns a BeautifulSoup() instance representing the HTML of the page."""

    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    return soup





soup=get_soup("https://letterboxd.com/film/sing-sing/")

def get_year(soup):
    match = re.search(r'\(([^)]+)\)',soup.title.string.strip())
    if match:
        year = match.group(1)
        return int(year)
    else:
        return None 






print(2023<=get_year(soup))








nominees=pd.read_csv("OscarWinners.csv")
def manual_url(movie_df): 
    um= movie_df.copy()  
    um=um[["Movie Title","Year"]].drop_duplicates()
    um.loc[:, '']='Check' 
    um["movie"] = um["Movie Title"].str.lower()
    um["movie"] = um["movie"].apply(lambda x: re.sub(r'[^a-zA-Z]', '-', str(x)))
    um["movie"] = um["movie"].apply(lambda x: re.sub(r'-+', '-', str(x)))
    um['url']='https://letterboxd.com/film/'+um["movie"]+'/'
    for idx, row in um.iterrows():
        try: 
            response=requests.get(row['url'])
            response.raise_for_status()
        except requests.exceptions.RequestException as e: 
            print(f"Failed: {e}")
        if response.status_code == 200: 
            year=get_year(BeautifulSoup(response.text, "html.parser"))
            if row['Year']-2<=year<=row['Year']: 
                um.loc[idx, 'url']='OK'
    
    return um






df_Movie=manual_url(nominees)

df_Movie.to_excel("Manual_Check.xlsx",index=False)







