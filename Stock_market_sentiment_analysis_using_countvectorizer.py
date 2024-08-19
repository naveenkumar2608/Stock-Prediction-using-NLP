'''
"Stock Sentiment Analysis Using News Headlines" is a project focused on predicting stock market trends
 by analyzing the sentiment of news headlines. The project involves collecting news headlines related to
 specific stocks, processing the text data to determine whether the sentiment is positive, negative, or
 neutral, and then correlating this sentiment with stock price movements. By leveraging natural language
 processing (NLP) techniques and machine learning models, the project aims to provide insights into how
public sentiment, as reflected in news headlines, can influence stock prices, aiding in investment decision-making.
'''


#Data Loading and Initial Exploration
import pandas as pd
df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")
df.head()

#Splits the data into training and testing sets based on the date.
# The training set includes data before January 1, 2015, while the test set includes data after December 31, 2014
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Removing punctuations.It removes all non-alphabetic characters from the text data, leaving only letters.
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)

#  Converts all the text in the headlines to lower case to maintain uniformity
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)

#Combining Headlines into a Single String
' '.join(str(x) for x in data.iloc[1,0:25])
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
headlines[0]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)

#Exploratory data analysis

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

