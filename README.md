<h1 align="center">Hollywood 5</h1>
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

- Due to computing limitations, we restrict our analysis to only best-picture awards. Though the models in this analysis could be easily extended to target other categories as well.
- Audience behavior may shift after nominations are announced, even before winners are revealed—introducing possible bias despite our review cutoff. We could implement some interaction term for reviews after to see if that helps our predicition at all.
- Our models are limited to the random sample of reviews, for the purpose of making it reasonable to process our data. With more time, or a more powerful machine, we could take the entire data set at face value and perhaps construct a more accurate profile of the public perception of the films.
- With our method, we do not take into account the sentiments of foreign-language reviewers, eliminating a significant (~30) percentage of our data set. Distinct national trends could affect the efficacy of our models.
- Audience behavior may shift after nominations are announced, even before winners are revealed—introducing possible bias despite our review cutoff.
- Letterboxd skews toward younger, internet-savvy users and may not reflect Academy preferences.
- Review volume varies across films, especially in smaller or foreign-language categories.
- NLP models may misinterpret sarcasm, humor, or inside references common in user-generated reviews.

<h2 align="center">The Model(s)</h2>

<h3>Gradient Boosting</h3>

Uses scikit learn's HistGradientBoostingClassifier to predict Best Picture winners based on text reviews. Gradient boosting builds a sequence of shallow decision trees, where each new tree tries to correct the mistakes of the previous ones. This approach is well-suited to high-dimensional data (thousands of text features) and can identify subtle signals in review language—such as combinations of words or phrases that may indicate stronger Oscar prospects.

In our context, movie reviews can contain complex patterns and nuance in language. Additionally, the structure or reviews can vary vastly between users. Thus Gradient Boosting is particularly effective in capturing these non-linear relationships in the data, allowing it to outperform simpler models like logistic regression.

The model is configured with parameters, such as max_depth, min_samples_split, and learning_rate, which help prevent overfitting. GridSerachCV was used to optimize hyperparameters, then best parameters were manually inserted into the model for ease of running. These hyperparameters are particularly useful in this context, where the dataset may contain high-dimensional features (e.g., words in reviews) that could lead to overfitting in simpler models. However, it is still prone to overfitting in our setting since there are not that many target years, even though there are many features (words).



---

<h2 align="center">Results </h2>

<h3>Gradient Boosting</h3>

<b>Years Correctly Predicted: 3</b>

Out of 11 years, the Gradient Boosting model predicted the Best Picture winner correctly in 3 years (27% accuracy). Even in years it missed the winner, the model still ranked the correct film within the top 3 contenders 4 of the remaining 8 years. While the exact prediction rate was modest, the model was consistently able to highlight strong candidates, offering valuable insights into potential Oscar winners.

---

<h2 align="center"> Discussion </h2>

---

<h2 align="center">Reproducing the Results</h2>

**The required Python packages are:**
- **pandas** – for data manipulation and analysis  
- **requests** – for making HTTP requests to fetch web pages  
- **beautifulsoup** – for parsing HTML and extracting data from web pages  
- **scikit-learn** – for machine learning models and data preprocessing
- **langdetect** - for identifying the language of film reviews
- **swifter** - for efficiently applying functions to a pandas dataframe
- **tqdm** - for tracking the progress of any number of processes

These can be installed using pip (or pip3):  

```bash
pip install pandas requests beautifulsoup4 scikit-learn langdetect swifter tqdm
```

### Running the Models

