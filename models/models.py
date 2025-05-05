import pandas as pd 
import os 
from isl2_neural import yearly_neural
from bagofwords_logit import bowlogit
from embedding_logit import embed_logit
from gradient_boosting import gb_main
from weighted_naive_bayes import weighted_naive_bayes
import os 

path = os.path.join("data", "best_picture_metadata_with_sampled_english_reviews.csv")

df=pd.read_csv(path)

bowlogit(df)

embed_logit()

weighted_naive_bayes(path)

yearly_neural(df,500)

gb_main(df)









