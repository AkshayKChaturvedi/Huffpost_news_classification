import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer


def clean_comment(comment):

    comment = re.sub(r"what's", "what is ", comment)

    comment = re.sub(r"\'ve", " have ", comment)

    comment = re.sub(r"can't", "cannot ", comment)

    comment = re.sub(r"n't", " not ", comment)

    comment = re.sub(r"i'm", "i am ", comment)

    comment = re.sub(r"\'re", " are ", comment)

    comment = re.sub(r"\'d", " would ", comment)

    comment = re.sub(r"\'ll", " will ", comment)

    comment = re.sub('\W', ' ', comment)

    comment = re.sub('\s+', ' ', comment)

    comment = comment.strip(' ')

    comment = comment.lower()

    return comment


def preprocess_comments(data):
    stop_words = set(stopwords.words('english'))

    stemmer = SnowballStemmer("english")

    documents = [" ".join([stemmer.stem(word) for word in clean_comment(comment).split(" ")
                           if word not in stop_words if word.isalnum()]) for comment in data]

    return documents


def split_data(x, y, test_size=0.30, random_state=2, **kwargs):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, **kwargs)

    return x_train, x_test, y_train, y_test


def generate_features(x_train, ngram_range=(1, 1), stop_words='english', strip_accents='unicode',
                      sublinear_tf=True, max_features=5000, **kwargs):
    tf_idf_vec = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words, strip_accents=strip_accents,
                                 sublinear_tf=sublinear_tf, max_features=max_features, **kwargs)

    tf_idf_vec.fit(x_train)

    train_tf_idf = tf_idf_vec.transform(x_train)

    return train_tf_idf, tf_idf_vec, tf_idf_vec.get_feature_names()


def model_fitting_and_get_training_accuracy(model, train_tf_idf, y_train, categories, base_classifier=None, **kwargs):
    if base_classifier:
        classifier = base_classifier(model(**kwargs))
    else:
        classifier = model(**kwargs)

    classifier.fit(train_tf_idf, y_train)

    predictions_train = classifier.predict(train_tf_idf)

    accuracy_train = accuracy_score(predictions_train, y_train)

    confusion_matrix_train_category = confusion_matrix(y_train, predictions_train)

    classification_report_train_category = classification_report(y_train, predictions_train, target_names=categories)

    return predictions_train, accuracy_train, confusion_matrix_train_category, classification_report_train_category, classifier


def get_test_accuracy(tf_idf_vec, x_test, classifier, y_test, categories):
    test_tf_idf = tf_idf_vec.transform(x_test)

    predictions_test = classifier.predict(test_tf_idf)

    accuracy_test = accuracy_score(predictions_test, y_test)

    confusion_matrix_test_category = confusion_matrix(y_test, predictions_test)

    classification_report_test_category = classification_report(y_test, predictions_test, target_names=categories)

    return predictions_test, accuracy_test, confusion_matrix_test_category, classification_report_test_category
