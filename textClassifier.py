import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('spamHamData.csv') # loads the data into a dataframe
print(df.head()) # prints the first 5 rows of the csv

print(df['Category'].value_counts()) # print the count of each category

x = df['Message'] # features (the messages)
y = df['Category'] # labels (the categories)
print(len(x)) # prints number of messages
