import re
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

df = pd.read_csv("LetterBox_URL.csv")
df = df.drop_duplicates(subset="url")
df = df.dropna(subset=["url"])

def scrape_letterboxd(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        desc_tag = soup.find("meta", property="og:description")
        description = desc_tag["content"].strip() if desc_tag else ""

        genre_tags = soup.select("div.text-sluglist a[href*='/films/genre/']")
        genre = " ".join(tag.get_text(strip=True) for tag in genre_tags)

        cast_tags = soup.select("div.cast-list a[href*='/actor/']")
        cast = " ".join(tag.get_text(strip=True) for tag in cast_tags)

        studio_tags = soup.select("div.text-sluglist a[href*='/studio/']")
        studios = " ".join(tag.get_text(strip=True) for tag in studio_tags)

        return (url, description, genre, cast, studios)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return (url, "", "", "", "", "", "", "")

def clean_text(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[,"\n\r“”‘’\'’]', '', s).strip()

df["Nominee Name(s)"] = df["Nominee Name(s)"].apply(clean_awardees)

results = []
with ThreadPoolExecutor(max_workers=30) as executor:
    future_to_url = {executor.submit(scrape_letterboxd, url): url for url in df["url"]}
    for future in as_completed(future_to_url):
        results.append(future.result())

metadata = {
    url: (desc, genre, cast, studios)
    for url, desc, genre, cast, studios in results
}

df["Description"] = df["url"].map(lambda u: metadata[u][0])
df["Genre"] = df["url"].map(lambda u: metadata[u][1])
df["Cast"] = df["url"].map(lambda u: metadata[u][2])
df["Studios"] = df["url"].map(lambda u: metadata[u][3])

for col in ["Description", "Genre", "Cast", "Studios"]:
    df[col] = df[col].apply(clean_text)

df = df.drop(columns=["Award Title", "Won", "Year", "url", "Nominee Name(s)"])

df.to_csv("film_metadata.csv", index=False)