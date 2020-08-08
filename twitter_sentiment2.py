import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

tweets_df = pd.read_csv('train.csv')
tweets_df = tweets_df.drop(['id'], axis = 1)
tweets_df['length'] = tweets_df['tweet'].apply(len)

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
tweets_countvectorizer = CountVectorizer(analyzer = message_cleaning, dtype = 'uint8').fit_transform(tweets_df['tweet'])
#print(tweets_countvectorizer.shape)
#print(vectorizer.get_feature_names())
#print(tweets_countvectorizer.toarray())
tweets = pd.DataFrame(tweets_countvectorizer.toarray())
X = tweets
y = tweets_df['label']
#print(x.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_predict_test))
