import re
import csv

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
    global raw_strings
    
    with open (filename,'r', encoding="utf-8") as csvfile:
        csvdata = csv.reader(csvfile)
        
        for row in csvdata:
            strings.append(row[3])
            labels.append(int(row[4]))
    
    strings_norm = []
    for example in strings :
        strings_norm.append(normalize(example))

    raw_strings = strings #save raw strings for output

    X_train, X_test, y_train, y_test = train_test_split(strings_norm,labels,test_size=0.10)



def analyze_multi():
    cv = CountVectorizer()
    X_train_count = cv.fit_transform(X_train)
    X_test_count = cv.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_count, y_train)
    prediction = model.predict(X_test_count)
    accuracy = model.score(X_test_count, y_test)

    return prediction, accuracy



def main():
    process_text('Training.csv')
    prediction_nb, accuracy_nb = analyze_multi()

    print('\n\n\nMULTINOMIAL NAIVE BAYES')
    for i in range (len(X_test)):
        print("\nTest data (raw):", raw_strings[i])
        print("Test data (normalized): ",X_test[i])
        print("Prediction: ", prediction_nb[i])
    print('\n\nAccuracy:', accuracy_nb)



if __name__=="__main__":
    main()