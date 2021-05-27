# import libraries
import pandas as pd
import numpy as np
import nltk
import sys
import warnings
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator
import pickle
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
warnings.simplefilter('ignore')
import subprocess
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('termcolor')
from termcolor import colored, cprint
nltk.download(['punkt','wordnet','stopwords'])

def load_data(database_filepath, table_name='messages'):
    """Load cleaned data from database into dataframe.
    Args:
        database_filepath: String. It contains cleaned data table.
        table_name: String. It contains cleaned data.
    Returns:
       X: numpy.ndarray. Disaster messages.
       Y: numpy.ndarray. Disaster categories for each messages.
       category_name: list. Disaster category names.
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, con=engine)

    category_names = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df.drop(['id','message','original','genre'], axis=1)

    return X, y, category_names


def tokenize(text, lemmatizer=WordNetLemmatizer()):
    """Tokenize text (a disaster message).
    Args:
        text: String. A disaster message.
        lemmatizer: nltk.stem.Lemmatizer.
    Returns:
        list. It contains tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def save_stats(X, Y, category_names, vocabulary_stats_filepath, category_stats_filepath):
    """Save stats
    Args;
        X: numpy.ndarray. Disaster messages.
        Y: numpy.ndarray. Disaster categories for each messages.
        category_names: Disaster category names.
        vocaburary_stats_filepath: String. Vocaburary stats is saved as pickel into this file.
        category_stats_filepath: String. Category stats is saved as pickel into this file.
    """
    # Check vocabulary
    vect = CountVectorizer(tokenizer=tokenize)
    X_vectorized = vect.fit_transform(X)

    # Convert vocabulary into pandas.dataframe
    keys, values = [], []
    for k, v in vect.vocabulary_.items():
        keys.append(k)
        values.append(v)
    vocabulary_df = pd.DataFrame.from_dict({'words': keys, 'counts': values})

    # Vocabulary stats
    vocabulary_df = vocabulary_df.sample(30, random_state=72).sort_values('counts', ascending=False)
    vocabulary_counts = list(vocabulary_df['counts'])
    vocabulary_words = list(vocabulary_df['words'])

    # Save vocaburaly stats
    with open(vocabulary_stats_filepath, 'wb') as vocabulary_stats_file:
        pickle.dump((vocabulary_counts, vocabulary_words), vocabulary_stats_file)

    # Category stats
    category_counts = list(Y.sum(axis=0))

    # Save category stats
    with open(category_stats_filepath, 'wb') as category_stats_file:
        pickle.dump((category_counts, list(category_names)), category_stats_file)


def build_model():
    """Build model.
    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    pipeline =Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),  ('tfidf', TfidfTransformer()),
                        ('clf',      MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))])
    # Set parameters for gird search
    ''' parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}
     # Set grid search
     cv = GridSearchCV(pipeline, param_grid=parameters)'''
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """Evaluate model
    Args:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    # Predict categories of messages.
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\nReport for the column ({}):\n'.format(colored(col, 'red', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))
        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
             print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))


def save_model(model, model_filepath):
    """Save model
    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_filepath: String. Trained model is saved as pickel into this file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 5:
        database_filepath, model_filepath, vocabulary_filepath, category_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        print('Saving stats...')
        save_stats(X, Y, category_names, vocabulary_filepath, category_filepath)

        print('Building model...')
        model = build_model()

        print('Training model...')
        # Ignoring UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no true samples.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl vocabulary_stats.pkl category_stats_pkl')


if __name__ == '__main__':
    main()