<h1 align="center">Hollywood 5 (potentially CineClassifiers or Film Forecasters or Red Carpet Recommenders? eh? eh? get it?)</h1>
<h2 align="center">By Orlando Di Leo, Matias Ibarburu, Luke McDonald, Daniel Urrutia, Jisoo Yoo</h2>
<p align="center"> <img src="https://hips.hearstapps.com/hmg-prod/images/overview-of-the-oscar-statue-at-meet-the-oscars-at-the-time-news-photo-1588178852.jpg" alt="An image of an Oscar trophy." width="500"> </p>
<h2 align="center">Project Description</h2>

<p align="justify">
Our goal is to build a model that predicts whether an Oscar-nominated film will win Best Picture based on audience sentiment and engagement. We use data from Letterboxd reviews to extract patterns in how viewers talk about nominated films. The hypothesis is that emotional tone, review volume, and fan engagement — combined with basic film metadata — can reveal subtle signals that align with Oscar outcomes. By focusing on reviews published only before each year's ceremony, we aim to simulate a real-world prediction setting.
</p>

---

<h2 align="center">The Data</h2>

<h3 align="left">Sources</h3>

- Our **Oscars award data** was manually compiled from the <a href="https://www.oscars.org">official Academy Awards website</a> for the years 2015-2025, which records the title of each nominee and the eventual winners.
- The **Viewer reviews and film metadata** were scraped from <a href="https://letterboxd.com/films/">Letterboxd.com</a>, a film-reviewing site similar to Goodreads. It also has an info page for each film with metadata that we scraped and compiled.
---
<h3 align="left">Features</h3>
<ul>
  <li>Description and genre of film, taken from Letterboxd</li>
  <li>Combined text of 10,000 reviews per Best Picture nominee (for NLP analysis)</li>
  <li>List of cast members and studios (as categorical features)</li>
  <li>Binary outcome (1 = winner, 0 = nominee that did not win)</li>
</ul>

<h3 align="left">Collection Methodology</h3>

<ol>
  <li>First, we scraped the nominees for Best Picture from 2015 to 2025 from the official Academy Awards website, encoding whether they won or not in our binary outcome variable. We also scraped the Oscars ceremony date for each year, so that we can filter out post-ceremony reviews later.</li>
  <li>Then, using our knowledge of Letterboxd's standard formatting, we added the Letterboxd URLs for each film into a data set to use for our scraper.</li>
  <li>LUKE EXPLAIN HOW YOU SCRAPED REVIEWS PER RATING</li>
  <li>Using a <code>requests</code> scraper, we then gathered film metadata such as the description text, the genre, cast names, and studio names.</li>
<li>
  We then combined the data from steps (3) and (4) into a master data set. This involved standardizing the "time posted" data for the reviews, filtering for reviews posted before the given year's ceremony date, and combining 10,000 randomly selected English-language reviews into a single text parameter for NLP processing.
  <div style="margin-left: 2em;">
    - At first, we tried taking the random sample before actually translating non-English reviews, but we lacked the processing power to pull it off.
  </div>
  <div style="margin-left: 2em;">
    - We then pivoted to the approach of using <code>langdetect</code> in conjunction with <code>swifter</code> to simply filter out non-English reviews <em>before</em> taking our sample, massively simplifying the process and reducing our runtime.
  </div>
</li>

  <li>MODEL DESIGNERS EXPLAIN THE PROCESS HERE</li>
</ol>

<h3>Limitations</h3>

- Our models are limited to the random sample of reviews, for the purpose of making it reasonable to process our data. With more time, or a more powerful machine, we could take the entire data set at face value and perhaps construct a more accurate profile of the public perception of the films.
- With our method, we do not take into account the sentiments of foreign-language reviewers, eliminating a significant (~30) percentage of our data set. Distinct national trends could affect the efficacy of our models.
- Audience behavior may shift after nominations are announced, even before winners are revealed—introducing possible bias despite our review cutoff.
- Letterboxd skews toward younger, internet-savvy users and may not reflect Academy preferences.
- Review volume varies across films, especially in smaller or foreign-language categories.
- NLP models may misinterpret sarcasm, humor, or inside references common in user-generated reviews.

---

<h2 align="center">The Model(s)</h2>

We use a classification approach to predict whether a nominated film won its Oscar. Initially, we apply NLP techniques to the review text and use classifiers such as a multilayer perceptron (MLP). The model learns patterns in both the language and numerical features of engagement.

Future extensions will include experiments with:
- TF-IDF vs. word embeddings
- Ensemble models (e.g. combining numeric + text pipelines)
- Incorporating prior nomination history or director recognition

---

<h2 align="center">Results and Recommendations</h2>

Preliminary results show that language and fan engagement can help distinguish Oscar winners from other nominees. While our full metrics are still being finalized, early model accuracy and precision outperform basic baselines. Certain categories (e.g. Best Picture, Acting Awards) show stronger signals than others.

We recommend future versions include historical Oscar trend features, more granular text sentiment scoring, and possibly external critic sources to supplement fan reviews.

---

<h2 align="center">Reproducing the Results</h2>

To reproduce the results:
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
