import pandas as pd
from pathlib import Path
from datetime import datetime

base_dir = Path(__file__).parent
data_dir = base_dir / "data"

print("Loading data")
reviews_df = pd.read_csv(
    data_dir / "letterboxd_combined_reviews_2015_2025.csv", low_memory=False
)
metadata_df = pd.read_csv(data_dir / "film_metadata.csv")
oscars_df = pd.read_csv(data_dir / "OscarWinners.csv")

print("Filtering Best Picture nominees")
best_picture_df = oscars_df[oscars_df["Award Title"] == "Best Picture"].copy()

best_picture_df["Ceremony Date"] = pd.to_datetime(
    best_picture_df["Ceremony Date"], errors="coerce"
).dt.tz_localize("UTC")

parsed = pd.to_datetime(reviews_df["date"], errors="coerce", utc=True)
mask_failed = parsed.isna() & reviews_df["date"].notna()
fallback = pd.to_datetime(
    reviews_df.loc[mask_failed, "date"], format="%d %b %Y", errors="coerce"
).dt.tz_localize("UTC")
parsed.loc[mask_failed] = fallback
reviews_df["parsed_date"] = parsed

merged_reviews = reviews_df.merge(
    best_picture_df[["Movie Title", "Ceremony Date"]],
    left_on="film_name",
    right_on="Movie Title",
    how="inner",
)

filtered_reviews = merged_reviews[
    merged_reviews["parsed_date"] <= merged_reviews["Ceremony Date"]
].dropna(subset=["review_text"])

print("Sampling up to 100 reviews per film")
sampled_reviews = (
    filtered_reviews.groupby("film_name")
    .apply(lambda g: g.sample(n=min(len(g), 100), random_state=42))
    .reset_index(drop=True)
)

merged_metadata = best_picture_df.merge(metadata_df, on="Movie Title", how="left")
expanded = sampled_reviews.merge(
    merged_metadata, left_on="film_name", right_on="Movie Title", how="left"
)

final_df = expanded[
    [
        "film_name",
        "Oscars Year",
        "Won",
        "Description",
        "Genre",
        "Cast",
        "Studios",
        "review_text",
    ]
].rename(
    columns={
        "film_name": "Film Name",
        "Oscars Year": "Year Nominated",
        "review_text": "Review Text",
    }
)

output_file = data_dir / "bp_embedding_data_100.csv"
final_df.to_csv(output_file, index=False)

print(f"CSV written to: {output_file}")
