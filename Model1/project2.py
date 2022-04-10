
#import libraries

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score

#import the dataset

dataset=pd.read_csv("restaurant_reviews.tsv",delimiter="\t",quoting=3)

import re
import nltk

#cleaning the dataset

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]).lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review=[ps.stem(word) for word in review ]#if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)
print(corpus) 


#create bag of words

cv=CountVectorizer(max_features=1500)

x=cv.fit_transform(corpus).toarray()

y=dataset.iloc[: ,-1].values
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#train the model using naive bayes

gnb=GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)

#confusion matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)

accuracy=accuracy_score(y_test,y_pred)

print(accuracy)