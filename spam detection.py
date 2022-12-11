#!/usr/bin/env python
# coding: utf-8

import re
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB


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

    X_train, X_test, y_train, y_test = train_test_split(strings_norm,labels,test_size=0.10)



def analyze_multi():
    cv = CountVectorizer()
    X_train_count = cv.fit_transform(X_train)
    X_test_count = cv.transform(X_test)

    model_nb = MultinomialNB()
    model_nb.fit(X_train_count, y_train)
    prediction_nb = model_nb.predict(X_test_count)
    accuracy_nb = model_nb.score(X_test_count, y_test)

    return prediction_nb, accuracy_nb



def analyze_gauss():
    cv = CountVectorizer()
    X_train_count = cv.fit_transform(X_train)
    X_train_count = X_train_count.toarray()
    X_test_count = cv.transform(X_test)
    X_test_count = X_test_count.toarray()

    model_g = GaussianNB()
    model_g.fit(X_train_count, y_train)
    prediction_g = model_g.predict(X_test_count)
    accuracy_g = model_g.score(X_test_count, y_test)

    return prediction_g, accuracy_g



def main():
    process_text('Training.csv')
    prediction_nb, accuracy_nb = analyze_multi()
    #prediction_g, accuracy_g = analyze_gauss()

    print('\n\n\nMULTINOMIAL NAIVE BAYES')
    print('\nTest data + prediction: ')
    for i in range (len(X_test)):
        print(X_test[i], "| ", prediction_nb[i])
        
    print('\nAccuracy:', accuracy_nb)

'''
    print('\n\n\nGAUSSIAN NAIVE BAYES')
    print('\nPrediction:', prediction_g)
    print('\nAccuracy:', accuracy_g)
'''

if __name__=="__main__":
    main()