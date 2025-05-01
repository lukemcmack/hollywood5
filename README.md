<h1 align="center">Hollywood 5 (potentially CineClassifiers or Film Forecasters or Red Carpet Recommenders? eh? eh? get it?)</h1>
<h2 align="center">By Orlando Di Leo, Matias Ibarburu, Luke McDonald, Daniel Urrutia, Jisoo Yoo</h2>
<p align="center"> <img src="https://hips.hearstapps.com/hmg-prod/images/overview-of-the-oscar-statue-at-meet-the-oscars-at-the-time-news-photo-1588178852.jpg" alt="An image of an Oscar trophy." width="500"> </p>
<h2 align="center">Project Description</h2>
<p align="justify">
The goal of our project is to predict whether an Oscar-nominated film will win its category based on audience sentiment. To do this, we use (NLP? or other) techniques on Letterboxd reviews to extract any meaningful patterns from how people talk about each film. Our hypothesis is that certain language cues and emotional signals - combined with metadata like review amount, average rating, and director identity - can help us distinguish likely winners from other nominees.
</p>

<h2 align="center">The Data</h2>

<h3 align="left">Sources</h3>
<ul>
  <li><b>Oscar Award Data:</b> Manually compiled from the official Oscars website, including award categories, nominated films, credited individuals, and winners from 2012 to 2025: https://www.oscars.org/oscars/ceremonies/ (LUKE McDONALD + DANIEL URRUTIA).</li>
  <li><b>Letterboxd Reviews:</b> User reviews and metadata (e.g., star rating, number of likes) scraped from Letterboxd using a custom-built crawler (LUKE McDONALD).</li>
</ul>

<h3 align="left">Features</h3>
<ul>
  <li>Review text</li>
  <li>Sentiment wording scores</li>
  <li>Average star rating, total number of reviews, and engagement (likes, comments)</li>
  <li>Binary outcome variable indicating Oscar win (1) or not (0)</li>
</ul>


<h3 align="left">Collection Methodology</h3>
<p>
We collected Letterboxd reviews for nominated films, filtering to include only reviews posted before the official Oscar award announcements. This allows our model to simulate a true prediction setting, using only information available prior to the results. Reviews were gathered across multiple years, with metadata including timestamps, review text, star ratings, and engagement metrics.
</p>

<h3 align="left">Limitations</h3>
<ul>
    <li>There may be bias in review behavior following the nominee announcements, as users may revise opinions or engage more with nominated films. While we excluded reviews posted after the award announcements, some noise could still be introduced during the nominee-to-winner period. Additionally, Letterboxd audiences may not be representative of Academy voters, and review volume can vary widely across films and years.</li>
    <li>Letterboxd reviews may be biased toward younger audiences, given that the app is relatively new, which may not reflect Academy voting trends.</li>
    <li>Some nominated films have relatively few reviews, which can affect model consistency.</li>
    <li>Review text may mention spoilers, sarcasm, jokes, or double negatives that introduce noise into features that will be used to predict classification (models used?).</li>
</ul>

<h2 align="center">The Model(s)</h2>

<p>
Our primary modeling approach classifies whether a nominated film won its Oscar category. We preprocess the Letterboxd review text features and feed this into a classifier. Then, the model learns patterns in the language and review engagement surrounding films to predict Oscar outcomes.
<p>
We plan to experiment with other models (models used?) for both text features and non-text data (e.g., star ratings, review counts, genre tags).
</p>

<h2 align="center">Results and Recommendations</h2>

<p>
So far, our model shows (describe) results in distinguishing between Oscar winners and non-winners using Letterboxd reviews. While exact performance metrics are still in development, initial classification accuracy and precision suggest that online audience response captures meaningful signals about industry outcomes.
</p>

<p>
We recommend future iterations explore ensemble models combining linguistic, numeric, and historical award trend features. Expanding the dataset across more years may also improve generalizability.
</p>

<h2 align="center">Reproducing the Results</h2>
<p>
To reproduce our results, clone this repository and install the required dependencies listed in <code>requirements.txt</code>. Then, run the pipeline scripts, starting with data collection and processing and ending with model training and evaluation.
</p>


<h3 align="left">Sources</h3>
<ul>
  <li><b>Oscar Nominee and Winner Data:</b> Manually compiled from the official <a href="https://www.oscars.org/">Academy Awards</a> website for years 2012–2025.</li>
  <li><b>Letterboxd Review Data:</b> Scraped using custom scripts built by the team, focused on reviews published before the Oscar ceremony each year.</li>
</ul>
