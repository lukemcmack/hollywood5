import pandas as pd
import requests
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import re
import unicodedata
import threading
import os

"""This file is not required for scraping, but is useful to see if your urls are pointing correctly quickly"""

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


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

    for char in ["'", ":", ",", ".", "!", "?", "&", "/", "(", ")"]:
        slug = slug.replace(char, "")

    slug = slug.replace(" ", "-")
    return slug


def check_url_exists(url):
    """Check if URL exists with proper headers"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        return requests.head(url, headers=headers, timeout=10).status_code == 200
    except:
        return False


def generate_slug_variations(base_slug, year):
    """Generate variations in Letterboxd's preferred order"""
    if pd.isna(year) or not str(year).isdigit():
        return [base_slug]

    year = int(year)
    variations = []

    variations.append(f"{base_slug}-{year}")

    for i in range(1, 3):
        variations.extend([f"{base_slug}-{year+i}", f"{base_slug}-{year-i}"])

    variations.append(base_slug)

    seen = set()
    return [x for x in variations if not (x in seen or seen.add(x))]


def process_movie(row, title_col, year_col):
    """Process a single movie to find its URL"""
    try:
        title = row[title_col]
        year = row[year_col]

        if pd.isna(title):
            return {
                "original_title": None,
                "year": year,
                "slug": None,
                "letterboxd_url": None,
            }

        base_slug = clean_title_to_slug(title)

        for slug in generate_slug_variations(base_slug, year):
            url = urljoin("https://letterboxd.com/film/", f"{slug}/")
            if check_url_exists(url):
                safe_print(f"Found: {title} ({year}) -> {url}")
                return {
                    "original_title": title,
                    "year": year,
                    "slug": slug,
                    "letterboxd_url": url,
                }

        safe_print(f"Not found: {title} ({year})")
        return {
            "original_title": title,
            "year": year,
            "slug": None,
            "letterboxd_url": None,
        }
    except Exception as e:
        safe_print(f"Error processing {row.get(title_col, '')}: {str(e)}")
        return {
            "original_title": row.get(title_col),
            "year": row.get(year_col),
            "slug": None,
            "letterboxd_url": None,
        }


def process_movies(
    input_csv, output_csv, title_col="Movie Title", year_col="Year", workers=10
):
    """Process all movies in parallel"""
    df = pd.read_csv(input_csv, keep_default_na=True, na_values=[""])

    df[title_col] = df[title_col].astype(str)

    df = df[~df[title_col].isin(["nan", "None", ""])]

    df = df.drop_duplicates([title_col, year_col])

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_movie, row, title_col, year_col)
            for _, row in df.iterrows()
        ]

        for future in futures:
            results.append(future.result())

    pd.DataFrame(results).to_csv(output_csv, index=False)
    safe_print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    process_movies(
        input_csv = os.path.join("data", "OscarWinners.csv"),
        output_csv="letterboxd_urls.csv",
    )
