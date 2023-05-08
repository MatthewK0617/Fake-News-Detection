import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split 
# https://www.sharpsightlabs.com/blog/scikit-train_test_split/
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv('/Users/matthewkim/Downloads/news.csv')

print(df.shape)
# print(df.head())

labels = df.label
print(labels.head())

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7) 
# x_train is initial set
# y_train are the labels of x_train (the same for x_test and y_test)

# make own custom vectorizer
## check their formulation and come up with own

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.4)
# stop_words is currently set to a predefined list of common words that can be ignored; max_df ignores terms 
## with higher frequency than 0.7
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# how relevant a word is to the passage: This is done by multiplying two metrics: how many times a word 
# appears in a document, and the inverse document frequency of the word across a set of documents. 

tfidf_train = tfidf_vectorizer.fit_transform(x_train) # combination of fit and transform - 
## generates learning model parameters and then applies on model to generate transformed data set
### turning into true or false?

# print(tfidf_train)
tfidf_test = tfidf_vectorizer.transform(x_test) # applies new tfidf_vectorizer parameters on model to transform x_test
# print(x_test)
# print(tfidf_test)

# look into the transformation process

# review below
pac = PassiveAggressiveClassifier(max_iter=50)
# https://www.youtube.com/watch?v=TJU8NfDdqNQ 
pac.fit(tfidf_train, y_train) # trying to get from x --> y. updates the pac model parameters
# how does this fitting work?
# black box regression?

print(y_train)

y_pred = pac.predict(tfidf_test) # predicting based on model parameters. generates labels based on parameters. 
# could be above regression line is true below is false (or more possibly, calculating based on distance from point)
score = accuracy_score(y_test, y_pred) # y_test are the correct labels 

print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])