# import libraries
import pandas
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def preprocess():
    # making a readable dataset
    all_text_reviews_dataframe = pandas.read_csv('yelp_labelled.txt', sep='\t')
    all_text_reviews_dataframe.columns = ['Review', 'Sentiment']
    array_of_reviews = []
    print(all_text_reviews_dataframe['Review'].size)
    for i in range(0, all_text_reviews_dataframe['Review'].size):
        text_review = re.sub('[^a-zA-Z]', ' ', all_text_reviews_dataframe['Review'][i])
        text_review = text_review.lower()
        text_review = text_review.split()
        porter_stemmer = nltk.stem.PorterStemmer()
        text_review = [porter_stemmer.stem(word) for word in text_review if not word in set(stopwords.words('english'))]
        text_review = ' '.join(text_review)
        array_of_reviews.append(text_review)

    # bag of words
    count_vectorizer = CountVectorizer(max_features=1500)
    X = count_vectorizer.fit_transform(array_of_reviews).toarray()
    y = all_text_reviews_dataframe.iloc[:, 1].values  # all rows and column index 1

    # training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

    # fitting naive bayes to set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # predicting
    y_pred = classifier.predict(X_test)

    # confusion matrix
    cf = confusion_matrix(y_pred, y_test)

    print(cf)

preprocessed_data = preprocess()

a = preprocessed_data
