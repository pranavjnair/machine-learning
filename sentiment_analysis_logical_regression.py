#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:13:13 2019

@author: Pranav Nair
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def calculate(message):
    # create data frame from yelp reviews
    print(message)
    dataset = pd.read_csv('yelp_labelled.txt', sep='\t', header=None)
    # print(dataset.head())
    dataset.columns = ['Text','Class']
    train_x = dataset['Text'].values
    train_y = dataset['Class'].values
    
    # Vectorizer to represent our data
    vectorizer = TfidfVectorizer()
    vectorized_train_x = vectorizer.fit_transform(train_x)
    print("vectorized_train_x shape::{0}".format(vectorized_train_x.shape))
    
    # Create logistic regression model
    print(train_x)
    print(vectorized_train_x)
    classifier = LogisticRegression()
    classifier.fit(vectorized_train_x, train_y)
    
    # Use test message and convert to vector representation
    sentence = [message]
    vectorized_message = vectorizer.transform(sentence)
    prediction = classifier.predict_proba(vectorized_message)
    
    # Get probabilities
    negative_score = prediction[0][0]
    positive_score = prediction[0][1]
    
    if negative_score >= .33 and positive_score >= .33:
        sentiment = 'neutral'
    elif negative_score > .66:
        sentiment = 'negative'
    else:
        sentiment = 'positive'
    
    print("Sentiment: {0}".format(sentiment))
    
    print("Negative score: {0}, Positive Score {1}".format(negative_score, positive_score))


#message = "Can I go negative on Workday for vacation days?"
message = "Very slow service"

calculate(message)



