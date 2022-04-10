
#import libraries

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

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

#feature extraction using TFidf

from sklearn.feature_extraction.text import TFidfVectorizer

tf=TFidfVectorizer(max_features=2500)
x=tf.fit_transform(corpus).toarray()
y=dataset.iloc[: ,-1].values

#create bag of words


print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#model training using SVM

from sklearn.svm import SVC

model= SVC()

model.fit(x_train,y_train)

y_pred1=model.predict(x_test)

print("training accuracy using SVM :" , model.score(x_train,y_train))
print("testing accuracy using SVM: ",model.score(x_test,y_test))

print("confusion matix by SVM")

cm=confusion_matrix(y_test,y_pred)
disp=plot_confusion_matrix(gnb,x_test,y_test,cmap=plt.cm.Blues)



print(cm)

accuracy=accuracy_score(y_test,y_pred)

print(accuracy)