**Stock Sentiment Analysis Using News Headlines**

**ABSTRACT :**
                Stock Sentiment Analysis Using News Headlines" is a project focused on predicting stock market trends by analyzing the sentiment of news headlines. The project involves collecting news headlines related to specific stocks, processing the text data to determine whether the sentiment is positive, negative, or neutral, and then correlating this sentiment with stock price movements. By leveraging natural language processing (NLP) techniques and machine learning models, the project aims to provide insights into how public sentiment, as reflected in news headlines, can influence stock prices, aiding in investment decision-making.

**DATASET :**
            
              The data set consist of 25 headlines for each day. The number of data in the dataset are 4102 . It has special characters in it . It is labelled as 0s and 1s. The 1 represent the positive word and 0 represent the negative word.  


 


Data Loading and Initial Exploration:
The project begins by loading and exploring the data:
1.	Data Loading:
o	The dataset (Data.csv) is loaded using pandas. The data is encoded in "ISO-8859-1" format to ensure compatibility.
o	A preview of the dataset is displayed using df.head().
2.	Data Splitting:
o	The dataset is split into a training set (containing data before January 1, 2015) and a test set (containing data after December 31, 2014). This split ensures that the model is trained on historical data and tested on future data.
3.	Data Preprocessing:
o	Non-alphabetic characters are removed from the text data to retain only letters.
o	Column names are renamed for easier access, and all text in the headlines is converted to lowercase to maintain uniformity.
4.	Combining Headlines:
o	Headlines from different columns are combined into a single string for each day, creating a list of combined headlines that will be used for further analysis.
Feature Extraction and Model Implementation:
1.	Bag of Words Implementation:
o	A CountVectorizer is used to implement the Bag of Words model, which converts the combined headlines into a matrix of token counts. The ngram_range is set to (2, 2) to capture bigrams (pairs of consecutive words).
2.	RandomForest Classifier:
o	A RandomForestClassifier is implemented with 200 estimators and the 'entropy' criterion. The model is trained on the transformed training dataset.
3.	Predicting Test Data:
o	The test dataset is transformed similarly to the training data, and predictions are made using the trained RandomForest model.
Evaluation of Model Performance:
1.	Accuracy and Classification Report:
o	The model's performance is evaluated using a confusion matrix, accuracy score, and classification report. These metrics provide insights into the model's ability to correctly classify positive and negative sentiments.
2.	Confusion Matrix Visualization:
o	A confusion matrix is plotted using matplotlib and seaborn. The matrix visually represents the number of correct and incorrect predictions for each class (Positive and Negative).

Executing using CountVectorizer:
## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)

OUTPUT:
 
  

Executing using TfidfVectorizer:
## implement BAG OF WORDS
tfidfvector=TfidfVectorizer(ngram_range=(2,2))
traindataset=tfidfvector.fit_transform(headlines)

 

 


Executing using Naïve Bayes:
# implement RandomForest Classifier
naive=MultinomialNB()
naive.fit(traindataset,train['Label'])

 
 

Conclusion:
                This project demonstrates the potential of using sentiment analysis on news headlines to predict stock market trends. By applying NLP techniques and machine learning models, the project successfully correlates public sentiment with stock price movements, providing valuable insights for investment decision-making.



