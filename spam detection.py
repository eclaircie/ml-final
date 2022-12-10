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
    
    # strip html tags, replace with ""
    example = re.sub('\<\w{1,2}\>',
                     '', example)
    
    # remove punctuation; 
    #replace with keyword " _PUNCT " (including spaces)
    example = re.sub('\W', ' ', example)
    
    # remove links, replace with keyword " _EXTERNALLINK "
    example = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)',
                     ' EXTERNALLINK ', example)

    # replace all excessive whitespace with a single space; 
    #(this will also normalize the punctuation & link counters above)
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

    X_train, X_test, y_train, y_test = train_test_split(strings,labels,test_size=0.1)


    return np.column_stack((strings_norm,labels))
    

    
#create dictionary of most common words from an array of strings, where spam = 1
#default dict size is 500 words
def create_dict(arr, dict_size=500):
    all_spam_words = [] #list of all UNIQUE spammy words
    
    for training_example in arr:
        #if message is SPAM
        if training_example[1]=='1':
            training_string = training_example[0] #get the string
            words = training_string.split() #split into words
            
            #for every word in string,
            for word in words:
                if not word in all_spam_words:
                    all_spam_words.append(word) #put word in temp if its not there already
    
    #count occurrence of word in all spammy strings
    spammy_strings = []
    
    for training_example in arr:
        #if message is SPAM
        if training_example[1]=='1':
            #store this string in a list to make it easier to count later
            spammy_strings.append(training_example[0])
            
            #training_string = training_example[0] #get the string
            #words = training_string.split() #split into words
    
    #convert spammystrings into one big string 
    spam_as_str = ''
    for element in spammy_strings:
        spam_as_str = spam_as_str + element
        
    #print (spam_as_str)
        
    #initialize counts as all unique spam words + all zeros
    counts = np.zeros(len(all_spam_words), )
    #counts = np.column_stack((all_spam_words, zer)) #dont need this???
    #print (all_spam_words,counts)
    
    # go through big string of spam, count each occurrence of each word in all_spam_words
    # add this occurence to counts
    
    spammy_words = [] #list of every word that appears in every spam 
    
    for i in spammy_strings:
        str_split = i.split()
        for j in str_split:
            spammy_words.append(j)

    
    
    
    for word in spammy_words:
        pos = all_spam_words.index(word) #pos of word in both all_spam_words AND counts

        #print (counts[pos])
        current_counts = counts[pos]
        counts[pos] = current_counts + 1

    # DEBUG ------------------
    #print(spam_as_str)
    print(np.shape(all_spam_words))
    for i in range (0,len(all_spam_words)):
        print (all_spam_words[i], counts[i])

    #for thing in spammy_words:
    #    print (thing)

    print(np.shape(all_spam_words))
    print(np.shape(counts))
    #dic = counts[:dict_size], #grab first (size) words from counts
    #return dic
    return np.array([all_spam_words, counts])
    


def main():
    training_set = process_text('Training.csv')


    dictionary = create_dict(training_set, 150) #150 words

    model = MultinomialNB()
    model.fit(dictionary, y_train)
    #make feature vectors for each comment, store in list, dictsize x no. of features



if __name__=="__main__":
    main()