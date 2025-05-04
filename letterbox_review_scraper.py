import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
BATCH_SIZE = 1
OUTPUT_FILE = "data/film_reviews_all_ratings.csv"
INPUT_CSV = "data/scraping_data_splits/LetterBox_URL_2012_2014.csv"

def parse_rating(rating_text):
    """Converts Letterboxd star ratings to numerical values."""
    star_map = {"★": 1, "½": 0.5}
    rating_value = 0
    for char in rating_text:
        rating_value += star_map.get(char, 0)
    return rating_value if rating_text else None

def fetch_with_retries(url, headers, max_retries=3, timeout=10):
    """Fetches a webpage with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response
            else:
                print(f"Attempt {attempt+1}: Received status code {response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}: {e} for {url}")
        
        time.sleep(2 ** attempt)  # Exponential backoff
    
    print(f"Max retries reached for {url}")
    return None

def get_unique_movies(csv_path):
    """Loads movies from CSV and returns only unique entries."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    
    if "film_name" not in df.columns or "url" not in df.columns:
        raise ValueError("CSV must contain 'film_name' and 'url' columns")
    
    # Remove duplicates based on URL (more reliable than film names)
    df_unique = df.drop_duplicates(subset=['url'], keep='first')
    
    print(f"Found {len(df)} total movies, {len(df_unique)} unique after removing duplicates")
    return df_unique[["film_name", "url"]].to_dict('records')

def scrape_film_reviews(film_info):
    """Scrapes reviews for a given film at all rating levels."""
    film_name = film_info["film_name"]
    base_url = film_info["url"].rstrip('/')  # Clean URL
    reviews = []
    
    # Generate all possible rating values from 0.5 to 5 in 0.5 increments
    rating_values = [round(i * 0.5, 1) for i in range(1, 11)]
    
    for rating in rating_values:
        
        for page in range(1, 257):  # Pages 1-256
            url = f"{base_url}/reviews/rated/{rating}/by/added-earliest/page/{page}/"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            response = fetch_with_retries(url, headers)
            if not response:
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            review_elements = soup.find_all("li", class_="film-detail")  # CORRECTED SELECTOR
            
            if not review_elements:
                break
            
            for review in review_elements:
                user_element = review.find("a", class_="avatar")
                username = user_element.get("href").split("/")[1] if user_element else "No username"
                
                rating_element = review.find("span", class_="rating")
                parsed_rating = parse_rating(rating_element.text.strip()) if rating_element else None
                
                review_text_element = review.find("div", class_="body-text")
                review_text = review_text_element.text.strip() if review_text_element else "No review text"
                
                date_element = review.find("span", class_="_nobr")
                time_element = review.find("time")
                date = time_element["datetime"] if time_element else (date_element.text.strip() if date_element else "No date")
                
                liked = review.find("span", class_="has-icon icon-16 icon-liked") is not None
                
                reviews.append({
                    "film_name": film_name,
                    "username": username,
                    "rating": parsed_rating,
                    "review_text": review_text,
                    "date": date,
                    "liked": liked,
                })
    
    print(f"Finished scraping {film_name} - collected {len(reviews)} reviews")
    return reviews

def save_to_csv(data):
    """Writes a batch of scraped reviews to the CSV file."""
    file_exists = os.path.isfile(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ["film_name", "username", "rating", "review_text", "date", "liked", "rating_filter", "source_url"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    # Load and verify unique movies
    film_info_list = get_unique_movies(INPUT_CSV)
    
    total_reviews = 0
    batch_reviews = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_film = {executor.submit(scrape_film_reviews, film_info): film_info for film_info in film_info_list}
        
        for future in as_completed(future_to_film):
            film_reviews = future.result()
            batch_reviews.extend(film_reviews)
            total_reviews += len(film_reviews)
            
            # Write when BATCH_SIZE is reached
            if len(batch_reviews) >= BATCH_SIZE:
                save_to_csv(batch_reviews)
                print(f"Batch saved. Total reviews written: {total_reviews}")
                batch_reviews = []

    # Write any remaining reviews
    if batch_reviews:
        save_to_csv(batch_reviews)
        print(f"Final batch saved. Total reviews written: {total_reviews}")

    print("\nScraping completed.")
    print(f"Total unique movies processed: {len(film_info_list)}")
    print(f"Total reviews collected: {total_reviews}")