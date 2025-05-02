import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent
data_dir = base_dir / "data"

reviews_df = pd.read_csv(data_dir / "letterboxd_combined_reviews_2015_2025.csv", low_memory=False)
metadata_df = pd.read_csv(data_dir / "film_metadata.csv")
oscars_df = pd.read_csv(data_dir / "OscarWinners.csv")

best_picture_df = oscars_df[oscars_df["Award Title"] == "Best Picture"]

merged_df = best_picture_df.merge(metadata_df, on="Movie Title", how="left")

review_groups = reviews_df.groupby("film_name")["review_text"].apply(
    lambda x: " ".join(str(r) for r in x if pd.notna(r))
).reset_index()
review_groups.rename(columns={"review_text": "Review Text"}, inplace=True)

final_df = merged_df.merge(review_groups, left_on="Movie Title", right_on="film_name", how="left")

final_df = final_df[[
    "Movie Title",
    "Year",
    "Description",
    "Genre",
    "Cast",
    "Studios",
    "Review Text"
]]
final_df.columns = [
    "Film Name",
    "Year Nominated",
    "Description",
    "Genre",
    "Cast",
    "Studios",
    "Review Text"
]

output_path = data_dir / "best_picture_metadata_with_reviews.csv"
final_df.to_csv(output_path, index=False)

print(f"Combined CSV written to: {output_path}")