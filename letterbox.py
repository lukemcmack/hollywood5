import requests
from bs4 import BeautifulSoup
from ftfy import fix_text
import pandas as pd
import re
import time
import random
import unicodedata


def get_soup(url):
    """Takes a URL and returns a BeautifulSoup() instance representing the HTML of the page."""

    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    return soup


def get_year(soup):
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
    """Remove accents while preserving all other characters"""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", input_str)
        if not unicodedata.combining(c)
    )


def clean_title_to_slug(title):
    """Precisely convert titles to Letterboxd-style slugs"""
    if pd.isna(title):
        return ""

    title = str(title)
    slug = title.lower()
    slug = remove_accents(slug)
    slug = re.sub(r"[^a-zA-Z0-9\s\-c]", "", slug)

    slug = slug.replace(" ", "-")
    slug = re.sub(r"-+", "-", slug)
    return slug


nominees = pd.read_csv("OscarWinners.csv")


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

df_Movie = pd.read_excel("Manual_Check2.xlsx")


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


movie_wy2 = manual_url_year2(df_Movie)

movie_wy2.to_excel("Manual_Check3.xlsx", index=False)
