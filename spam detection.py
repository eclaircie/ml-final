#!/usr/bin/env python
# coding: utf-8

import re
import csv
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def normalize(example):
    # make everything lowercase 
    example = example.lower()

    # remove punctuation
    example = re.sub('\W', ' ', example)

    # replace all excessive whitespace with a single space
    example = re.sub('\s{2,}',
                     ' ', example)

    return example



def process_text(filename):
    # read 4th col of file (data) + 5th col (label)
    strings =[]
    labels = []
    global X_train, X_test, y_train, y_test
    
    with open (filename,'r', encoding="utf-8") as csvfile:
        csvdata = csv.reader(csvfile)
        
        for row in csvdata:
            strings.append(row[3])
            labels.append(int(row[4]))
    
    strings_norm = []
    for example in strings :
        strings_norm.append(normalize(example))

    X_train, X_test, y_train, y_test = train_test_split(strings_norm,labels,test_size=0.15)


    return np.column_stack((strings_norm,labels))
    


def main():
    training_set = process_text('Training.csv')

    cv = CountVectorizer()
    X_train_count = cv.fit_transform(X_train)
    X_test_count = cv.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_count,y_train)
    prediction = model.predict(X_test_count)
    accuracy = model.score(X_test_count,y_test)

    print('TEST DATA:', X_test)
    print('PREDICTION:', prediction)
    print('ACCURACY:', accuracy)



if __name__=="__main__":
    main()