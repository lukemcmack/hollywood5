import requests
from bs4 import BeautifulSoup
from ftfy import fix_text
import pandas as pd
import re
import time
import unicodedata
import os

"""This script retreives the URLs for the nominated movies, and creates a list of movies that could not me found and need to be checked manually"""

def get_soup(url):
    """Fetches the HTML content of a URL and returns a BeautifulSoup object."""
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    return soup


def get_year(soup):
    """Extracts the year from the <title> tag of a BeautifulSoup object."""
    match = re.search(r"\((\d{4})\)", soup.title.string.strip())
    if match:
        year = match.group(1)
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None
    else:
        year = None
    return year


def remove_accents(input_str):
    """Remove accents while preserving all other characters."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", input_str)
        if not unicodedata.combining(c)
    )


def clean_title_to_slug(title):
    """Precisely convert titles to Letterboxd-style slugs."""
    if pd.isna(title):
        return ""

    title = str(title)
    slug = title.lower()
    slug = remove_accents(slug)
    slug = re.sub(r"[^a-zA-Z0-9\s\-c]", "", slug)
    slug = slug.replace(" ", "-")
    slug = re.sub(r"-+", "-", slug)
    return slug

def manual_url(movie_df):
    um = movie_df.copy()
    um = um[["Movie Title", "Year"]].drop_duplicates()
    um.loc[:, "Check"] = "Check"
    um["movie"] = um["Movie Title"].apply(lambda x: clean_title_to_slug(str(x)))
    um["url"] = "https://letterboxd.com/film/" + um["movie"] + "/"
    for idx, row in um.iterrows():
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(row["url"], headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed: {e}")
            um.loc[idx, "Check"] = e
            time.sleep(5)
        if response.status_code == 200:
            year = get_year(BeautifulSoup(response.text, "html.parser"))
            if (
                row["Year"] is not None
                and year is not None
                and row["Year"] - 2 <= year <= row["Year"]
            ):
                um.loc[idx, "Check"] = "OK"

    return um

def manual_url_year1(movie_df):
    um = movie_df.copy()
    um["year1"] = um["Year"] - 1
    um["year1"] = um["year1"].astype(str)
    for idx, row in um.iterrows():
        if row["Check"] != "OK":
            row["url"] = (
                "https://letterboxd.com/film/" + row["movie"] + "-" + row["year1"] + "/"
            )
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(row["url"], headers=headers, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Failed: {e}")
                um.loc[idx, "Check"] = e
                time.sleep(5)
            if response.status_code == 200:
                year = get_year(BeautifulSoup(response.text, "html.parser"))
                if (
                    row["Year"] is not None
                    and year is not None
                    and row["Year"] - 2 <= year <= row["Year"]
                ):
                    um.loc[idx, "Check"] = "OK"
                    um.loc[idx, "url"] = row["url"]

    return um


def manual_url_year2(movie_df):
    um = movie_df.copy()
    um["year2"] = um["Year"] - 2
    um["year2"] = um["year2"].astype(str)
    for idx, row in um.iterrows():
        if row["Check"] != "OK":
            row["url"] = (
                "https://letterboxd.com/film/" + row["movie"] + "-" + row["year2"] + "/"
            )
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(row["url"], headers=headers, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Failed: {e}")
                um.loc[idx, "Check"] = e
                time.sleep(5)
            if response.status_code == 200:
                year = get_year(BeautifulSoup(response.text, "html.parser"))
                if (
                    row["Year"] is not None
                    and year is not None
                    and row["Year"] - 2 <= year <= row["Year"]
                ):
                    um.loc[idx, "Check"] = "OK"
                    um.loc[idx, "url"] = row["url"]

    return um


# Define file paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")

csv_file = os.path.join(data_dir, "OscarWinners.csv")

# Load nominees dataset
nominees = pd.read_csv(csv_file)
movie_wy2=manual_url_year2(manual_url_year1(manual_url(nominees)))
movie_wy2.to_csv("Manual_Check.csv", index=False)

