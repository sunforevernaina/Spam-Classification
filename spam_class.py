import pandas as pd
import numpy as np
import re    #  re (regular expression)
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Reading data from file
messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

# Data pre-processing
nltk.download('stopwords')
ps = PorterStemmer()      # initializing variable for stemming
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    # removes special characters from the text
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # converting all words into their base form (stemming)
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=2500) 
# To get the no. of unique words we need in each column
X = cv.fit_transform(corpus).toarray()
# print(X.shape)
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values
# For getting rid of the dummy variable trap

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
print(conf_mat)
print(classification_report(y_test,y_pred))





