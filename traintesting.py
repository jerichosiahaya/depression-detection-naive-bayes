#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import string
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report


#import dataset
dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
dataset_encode = "ISO-8859-1"
data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding = dataset_encode, names = dataset_columns)

data.drop(['ids','date','flag','user'],axis = 1,inplace = True)

#remove punctuation
def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
data['clean_text']=data['text'].apply(lambda x: remove_punctuation(x))

#remove hyperlink
data['clean_text'] = data['clean_text'].str.replace(r"http\S+", "") 
#remove emoji
data['clean_text'] = data['clean_text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
#convert all words to lowercase
data['clean_text'] = data['clean_text'].str.lower()

#tokenization
nltk.download('punkt')
def tokenize(text):
    split=re.split("\W+",text) 
    return split
data['clean_text_tokenize']=data['clean_text'].apply(lambda x: tokenize(x.lower()))

#stopwords
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text
data['clean_text_tokenize_stopwords'] = data['clean_text_tokenize'].apply(lambda x: remove_stopwords(x))


# store label and text into new dataframe
new_data = pd.DataFrame()
new_data['text'] = data['clean_text']
new_data['label'] = data['target']
new_data['label'] = new_data['label'].replace(4,1) # 1 for positive, 0 for negative

#split training and testing
X = new_data['text']
y = new_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)


#### Modelling
# model pipeline tfidf -> multinomialnb
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#fitting train data to model
model.fit(X_train,y_train)
#predict with testing data
validation = model.predict(X_test)

from sklearn.metrics import accuracy_score
#accuracy score
print(accuracy_score(y_test, validation))
#classification report
print(classification_report(y_test, validation))