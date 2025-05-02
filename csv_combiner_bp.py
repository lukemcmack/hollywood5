import pandas as pd
from pathlib import Path
from datetime import datetime

base_dir = Path(__file__).parent
data_dir = base_dir / "data"

print("Loading CSVs")
reviews_df = pd.read_csv(data_dir / "letterboxd_combined_reviews_2015_2025.csv", low_memory=False)
metadata_df = pd.read_csv(data_dir / "film_metadata.csv")
oscars_df = pd.read_csv(data_dir / "OscarWinners.csv")

print("Filtering Best Picture nominees")
best_picture_df = oscars_df[oscars_df["Award Title"] == "Best Picture"].copy()

print("Parsing ceremony dates")
best_picture_df["Ceremony Date"] = pd.to_datetime(
    best_picture_df["Ceremony Date"], errors="coerce"
).dt.tz_localize("UTC")

print("Standardizing review dates")
parsed = pd.to_datetime(reviews_df["date"], errors="coerce", utc=True)

mask_failed = parsed.isna() & reviews_df["date"].notna()
fallback = pd.to_datetime(
    reviews_df.loc[mask_failed, "date"],
    format="%d %b %Y",
    errors="coerce"
).dt.tz_localize("UTC")

parsed.loc[mask_failed] = fallback
reviews_df["parsed_date"] = parsed

fallback = pd.to_datetime(
    reviews_df.loc[mask_failed, "date"],
    format="%d %b %Y",
    errors="coerce"
).dt.tz_localize("UTC")

reviews_df["parsed_date"] = parsed

print("Merging reviews with Oscar nominations") 
reviews_with_dates = reviews_df.merge(
    best_picture_df[["Movie Title", "Ceremony Date"]],
    left_on="film_name",
    right_on="Movie Title",
    how="inner"
)

print("Filtering reviews written on or before the ceremony")
filtered_reviews = reviews_with_dates[
    reviews_with_dates["parsed_date"] <= reviews_with_dates["Ceremony Date"]
]

print("Combining reviews per film")
review_groups = filtered_reviews.groupby("film_name")["review_text"].apply(
    lambda x: " ".join(str(r) for r in x if pd.notna(r))
).reset_index()
review_groups.rename(columns={"review_text": "Review Text"}, inplace=True)

print("Merging full metadata with review texts")
merged_df = best_picture_df.merge(metadata_df, on="Movie Title", how="left")
final_df = merged_df.merge(review_groups, left_on="Movie Title", right_on="film_name", how="left")

print("Renaming and finalizing columns")
final_df = final_df[[
    "Movie Title", "Oscars Year", "Won", "Description", "Genre", "Cast", "Studios", "Review Text"
]]
final_df.columns = [
    "Film Name", "Year Nominated", "Won", "Description", "Genre", "Cast", "Studios", "Review Text"
]

output_path = data_dir / "best_picture_metadata_with_reviews_filtered.csv"
final_df.to_csv(output_path, index=False)

print(f"CSV written to: {output_path}")
