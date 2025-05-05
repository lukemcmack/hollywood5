from isl2_neural import yearly_neural
import pandas as pd 

df=pd.read_csv("data/best_picture_metadata_with_sampled_english_reviews.csv")

#dataframe, year, and number of words in bag of words. Basic Neural Network
#basic_neural=neural(df,500)

yearly_neural(df,500)





