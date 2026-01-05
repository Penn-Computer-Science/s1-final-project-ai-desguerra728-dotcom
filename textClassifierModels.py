import sys
import pandas as pd
import numpy as np
import nltk
import sklearn
import matplotlib.pyplot as plt

# print versions of libraries
print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('NumPy: {}'.format(np.__version__))
print('NLTK: {}'.format(nltk.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print()

# load dataset
df = pd.read_csv('spamHamData.csv') # loads the data into a dataframe
print(df.head()) # prints the first 5 rows of the csv
print()

print(df['Category'].value_counts()) # print the count of each category
print()

x = df['Message'] # features (the messages)
print("UNPROCESSED MESSAGES:")
print(x[:10])
y = df['Category'] # labels (the categories)
print("Number of messages: " + str(len(x))) # prints number of messages
print()


# PREPROCESS TEXT DATA

# convert class labels to binary vals, spam=1, ham=0
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Encoded labels: " + str(y[:10])) # print first 10 encoded labels
print()

# AI assisted: inserting regular expressions and defining variable processed
processed = x
# replace URLs (matches http/https and www.* anywhere in text)
processed = processed.str.replace(r'(?:https?://\S+|www\.\S+)', 'webaddress', regex=True)
# replace numbers (integers and decimals)
processed = processed.str.replace(r'\d+(?:\.\d+)?', 'numbr', regex=True)
# remove punctuation (keep word chars and whitespace)
processed = processed.str.replace(r'[^\w\s]', ' ', regex=True)
# replace email addresses with 'emailaddr'
processed = processed.str.replace(r'\S+@\S+', 'emailaddr', regex=True)
# replace money symbols with 'moneysymb'
processed = processed.str.replace(r'£|\$', 'moneysymb', regex=True)
# replace 10 digit phone numbers with 'phonenumbr'
processed = processed.str.replace(r'\b\d{10}\b', 'phonenumbr', regex=True)
# replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ', regex=True)
# remove leading and trailing white whitespace
processed = processed.str.strip()
# convert to lowercase
processed = processed.str.lower()
print("PREPROCESSED MESSAGES:")
print(processed[:10]) # print first 10 processed messages
print()

#remove stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: " ".join(term for term in x.split() if term not in stop_words))

#remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

from nltk.tokenize import word_tokenize
# creating a bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)
print('number of words: {}'.format(len(all_words)))
print('15 most frequent words:{}'.format(all_words.most_common(15)))
print()

# use the 150 most common words as features
# features are the words in the messages
word_features = list(dict(all_words.most_common(150)).keys())


def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    
    return features

# example
print(processed[0])
features= find_features(processed[0])
for key,value in features.items():
    if value == True:
        print(key)

# find features for all messages
messages = zip(processed, y)

# define a seed for reproducibility
seed = 1
np.random.seed(seed)
# make messages a list so it can be shuffled/reused
messages = list(messages)
import random
random.seed(seed)
random.shuffle(messages)

# find features for each SMS messages
featuresets = [(find_features(text), label) for text, label in messages]

#  split training adn tetsing datta sets using sklearn
from sklearn import model_selection
train, test = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)

print("num of train: " + str(len(train)))
print("num of test: " + str(len(test)))
print()


# SCIKIT_LEARM CLASSIFIERS WITH NLTK
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# define models to train
names = ['K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SGD Classifier', 'Multinomial NB', 'SVC Linear']
classfiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classfiers))
print(models)

# wrap models in NLTK then train and test
from nltk.classify.scikitlearn import SklearnClassifier

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(train)
    accuracy = nltk.classify.accuracy(nltk_model, test)*100 # calculates accuracy (multiply by 100 to get percent)
    print('{}: Accuracy: {}'.format(name, accuracy))
    # make class lebel prediction for testing set
    txt_features, labels = zip(*test)

    prediction = nltk_model.classify_many(txt_features)

    # print a confusion matrix and classification report
    print(classification_report(labels, prediction))


    # AI assisted: plotting confusion matrix
    # build a confusion matrix DataFrame
    cm = confusion_matrix(labels, prediction)
    cm_df = pd.DataFrame(cm, index=['ham', 'spam'], columns=['ham', 'spam'])
    print(cm_df)

    # plot confusion matrix using matplotlib
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix - {}'.format(name))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['ham', 'spam'])
    ax.set_yticklabels(['ham', 'spam'])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


#  ensemble method - Voting Classifier
#  using several methods to vote whether a message is spam or ham
from sklearn.ensemble import VotingClassifier

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(train)
accuracy = nltk.classify.accuracy(nltk_ensemble, test)*100 # calculates accuracy (multiply by 100 to get percent)
print('Ensemble Method Accuracy: {}'.format(accuracy))

# make class lebel prediction for testing set
txt_features, labels = zip(*test)

prediction = nltk_ensemble.classify_many(txt_features)

# print a confusion matrix and classification report
print(classification_report(labels, prediction))


# AI assisted: plotting confusion matrix
# build a confusion matrix DataFrame
cm = confusion_matrix(labels, prediction)
cm_df = pd.DataFrame(cm, index=['ham', 'spam'], columns=['ham', 'spam'])
print(cm_df)

# plot confusion matrix using matplotlib
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_title('Confusion Matrix - Voting Classifier')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['ham', 'spam'])
ax.set_yticklabels(['ham', 'spam'])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
