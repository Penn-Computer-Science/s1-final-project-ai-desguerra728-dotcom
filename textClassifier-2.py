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
y = df['Category'] # labels (the categories)
print(len(x)) # prints number of messages


# PREPROCESS TEXT DATA

# convert class labels to binary vals, spam=1, ham=0
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y[:10]) # print first 10 encoded labels

processed = x.str.lower() # convert to lowercase

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: " ".join(term for term in x.split() if term not in stop_words))
