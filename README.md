<h1 align="center">Hollywood 5 (potentially CineClassifiers or Film Forecasters or Red Carpet Recommenders? eh? eh? get it?)</h1>
<h2 align="center">By Orlando Di Leo, Matias Ibarburu, Luke McDonald, Daniel Urrutia, Jisoo Yoo</h2>
<p align="center"> <img src="https://hips.hearstapps.com/hmg-prod/images/overview-of-the-oscar-statue-at-meet-the-oscars-at-the-time-news-photo-1588178852.jpg" alt="An image of an Oscar trophy." width="500"> </p>
<h2 align="center">🎬 Project Description</h2>

<p align="justify">
Our goal is to build a model that predicts whether an Oscar-nominated film will win its category, based on audience sentiment and engagement. We use data from Letterboxd reviews to extract patterns in how viewers talk about nominated films. The hypothesis is that emotional tone, review volume, and fan engagement — combined with basic film metadata — can reveal subtle signals that align with Oscar outcomes. By focusing on reviews published only before each year's ceremony, we aim to simulate a real-world prediction setting.
</p>

---

<h2 align="center">🗂️ The Data</h2>

<h3>Sources</h3>

- **Oscar Award Data**: Manually compiled from the official Academy Awards website (2015–2025), including award categories, nominee and winner names, and credited producers.  
  <i>(Data collection by Luke McDonald, Daniel, and Matias)</i>  
- **Letterboxd Reviews**: Scraped using a custom crawler, gathering user-written reviews, timestamps, star ratings, likes, and comment counts for each nominated film.  
  <i>(Scraping by Luke)</i>
- **Letterboxd Metadata**: Scraped using a custom crawler, gathering movie data such as description, producer, etc. for each nominated film.  
  <i>(Scraping by Jisoo)</i>
---
<h3>Features</h3>
<ul>
  <li>Preprocessed review text (for NLP analysis)</li>
  <li>Average star rating and number of reviews</li>
  <li>Engagement metrics (likes, comments)</li>
  <li>Director, editor, producer(s), and studio (as categorical features)</li>
  <li>Binary outcome (1 = winner, 0 = nominee that did not win)</li>
</ul>


<h3 align="left">Collection Methodology</h3>

<p>
We scraped Letterboxd reviews for each nominated film, filtering to include only reviews posted before the Oscar ceremony of that year. This simulates a true prediction setting based on pre-award public sentiment. We also manually compiled film-level metadata such as director, editor, producers, and production studio from official Oscar pages and IMDb. These categorical variables give us additional context about the film’s pedigree—information we suspect might carry predictive weight alongside audience reception.
</p>

<h3>Limitations</h3>

- Audience behavior may shift after nominations are announced, even before winners are revealed—introducing possible bias despite our review cutoff.
- Letterboxd skews toward younger, internet-savvy users and may not reflect Academy preferences.
- Review volume varies across films, especially in smaller or foreign-language categories.
- NLP models may misinterpret sarcasm, humor, or inside references common in user-generated reviews.
- Not all nominee categories are equally represented in the data (e.g. Short Films vs. Best Picture).

---

<h2 align="center">🧠 The Model(s)</h2>

We use a classification approach to predict whether a nominated film won its Oscar. Initially, we apply NLP techniques to the review text and use classifiers such as a multilayer perceptron (MLP). The model learns patterns in both the language and numerical features of engagement.

Future extensions will include experiments with:
- TF-IDF vs. word embeddings
- Ensemble models (e.g. combining numeric + text pipelines)
- Incorporating prior nomination history or director recognition

---

<h2 align="center">📊 Results and Recommendations</h2>

Preliminary results show that language and fan engagement can help distinguish Oscar winners from other nominees. While our full metrics are still being finalized, early model accuracy and precision outperform basic baselines. Certain categories (e.g. Best Picture, Acting Awards) show stronger signals than others.

We recommend future versions include historical Oscar trend features, more granular text sentiment scoring, and possibly external critic sources to supplement fan reviews.

---

<h2 align="center">🔁 Reproducing the Results</h2>

To reproduce the results:
1. Clone the repository.
2. Run the scraping scripts (if access is granted) or load the provided dataset.
3. Install dependencies from `requirements.txt`.
4. Run the pipeline scripts in order:
   - `data_cleaning.py`
   - `feature_engineering.py`
   - `model_training.py`
   - `evaluate_model.py`

This will allow you to fully replicate the modeling and evaluation steps described above.

---

<h3>📚 Sources</h3>

- **Oscar Nominee & Winner Data**: [Academy Awards Archive (2015–2025)](https://www.oscars.org/oscars/ceremonies/)
- **Letterboxd Review Data**: Scraped using custom Selenium + BeautifulSoup scripts (not for commercial use)

---