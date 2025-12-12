import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# pip install -U scikit-learn

df = pd.read_csv('spamHamData.csv') # loads the data into a dataframe
print(df.head()) # prints the first 5 rows of the csv

print(df['Category'].value_counts()) # print the count of each category

x = df['Message'] # features (the messages)
y = df['Category'] # labels (the categories)
print(len(x)) # prints number of messages

# splits the data into training and testing. 20% testing; 80% training
# rerunning the code will produce the same split because of random_state
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 22)

len(x_train) # print num of training messages

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,  classification_report

# create three different text classifiers
pipeMNB =Pipeline([
    ('tfidf', TfidfVectorizer(stop_words = 'english')),
    ('clf', MultinomialNB()),
])

pipeCNB = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,3))),
    ('clf', ComplementNB()),
])

pipeSVC = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,3))),
    ('clf', LinearSVC()),
])

pipeMNB.fit(x_train, y_train) # train MNB model
predictMNB = pipeMNB.predict(x_test) # MNB predicts on test data (testing)
accuracy_score(y_test, predictMNB) # accuracy of MNB model
print(f'MNB:{accuracy_score(y_test, predictMNB):.2f}') # print MNB accuracy to 2 decimal places

pipeCNB.fit(x_train, y_train) # train CNB model
predictCNB = pipeCNB.predict(x_test) # CNB predicts on test data (testing)
accuracy_score(y_test, predictCNB) # accuracy of CNB model
print(f'CNB:{accuracy_score(y_test, predictCNB):.2f}') # print CNB accuracy to 2 decimal places

pipeSVC.fit(x_train, y_train) # train SVC model
predictSVC = pipeSVC.predict(x_test) # SVC predicts on test data (testing)
accuracy_score(y_test, predictSVC) # accuracy of SVC model
print(f'SVC:{accuracy_score(y_test, predictSVC):.2f}') # print SVC accuracy to 2 decimal places

print(classification_report(y_test, predictSVC)) # print detailed classification report for SVC model


# TF-IDF (Term Frequency-Inverse Document Frequency): Weights words based on their importance in a document relative to the entire corpus.
# N-grams: Considers sequences of words (e.g., "bag of words") to capture some contextual information.

# MNB:0.95
# CNB:0.98
# SVC:0.99
