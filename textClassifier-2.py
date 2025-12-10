import sys
import pandas as pd
import numpy as np
import nltk
import sklearn

# print versions of libraries
print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('NumPy: {}'.format(np.__version__))
print('NLTK: {}'.format(nltk.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))

# load dataset
df = pd.read_csv('spamHamData.csv') # loads the data into a dataframe
print(df.head()) # prints the first 5 rows of the csv

print(df['Category'].value_counts()) # print the count of each category

x = df['Message'] # features (the messages)
print(x[:10])
y = df['Category'] # labels (the categories)
print(len(x)) # prints number of messages


# PREPROCESS TEXT DATA

# convert class labels to binary vals, spam=1, ham=0
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y[:10]) # print first 10 encoded labels

processed = x.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-z-A]{2,3}(/\S*)?$', 'webaddress')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
processed = processed.str.replace(r'[^\w\d\s]', ' ') # remove punctuation

print(processed[:10]) # print first 10 processed messages


# convert to lowercase
processed = processed.str.lower()

#remove stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: " ".join(term for term in x.split() if term not in stop_words))

#remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

from nltk.tokenize import word_tokenize
# Tokenization is the process of breaking down text into smaller units called tokens, which can be words, phrases, or sentences.
# creating a bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

print(words[:10]) # print first 10 tokens of last message
print(all_words[:10]) # print first 10 words in bag-of-words

all_words = nltk.FreqDist(all_words)
print('number of words: {}'.format(len(all_words)))
print('most frequent words:{}'.format(all_words.most_common(15)))

# 36:58