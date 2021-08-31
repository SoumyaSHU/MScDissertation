import re
import time
import joblib
import string
import nltk
import pandas as pd
import visualize_plots
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from visualize_plots import show_sentiments
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
import visualize_plots
from visualize_plots import get_heatmap, show_sentiments, roc_curves, report_classification, cf_matrix

def perform_train_test_split(df):
    X = df.drop(columns=['sentiment'])
    y = df['sentiment']
    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


def get_word_cloud(tweets):
    comment_words = ''
    for val in tweets['text']:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "

    wordcloud = WordCloud(width=800, height=800,
                          background_color='black',
                          min_font_size=10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_transformations():
    transformer = ColumnTransformer(
        transformers=[('tfidf', TfidfVectorizer(), 'text')],
        sparse_threshold=0
    )
    return transformer


def preprocess_tweet_text(tweet):
    # convert text to lower case
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    ps = PorterStemmer()
    filtered_words = [ps.stem(w) for w in filtered_words]
    return " ".join(filtered_words)


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


models = []
models.append(('logistic_regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('k_neighbours', KNeighborsClassifier()))
models.append(('decision_tree', DecisionTreeClassifier()))
models.append(('gaussian_naive_bayes', GaussianNB()))
models.append(('support_vector', SVC()))
models.append(('random_forest', RandomForestClassifier()))

param_grid = {
    'transformer': {
        'transformer__num__feature_range': [(0, 1)],
        'transformer__tfidf__max_features': [300],
        'transformer__tfidf__max_df': [1.0]
    },
    'k_neighbours': {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': [3, 5]
    },
    'logistic_regression': {
        'model': [LogisticRegression()],
        'model__C': [0.01, 0.1, 1.0]
    },
    'gaussian_naive_bayes': {
        'model': [GaussianNB()],
        'model__var_smoothing': [0.00000001, 0.000000001, 0.00000001]
    },
    'support_vector': {
        'model': [SVC()],
        # 'model__C': [1, 10, 100, 1000],
        'model__gamma': [0.001, 0.0001],
        'model__probability': [True]
    },
    'decision_tree': {
        'model': [DecisionTreeClassifier()],
        'model__max_depth': [1, 2, 3, 4, 5],
        'model__max_features': [None, "auto", "sqrt", "log2"]
    },
    'random_forest': {
        'model': [RandomForestClassifier()],
        'model__max_features': [3, 4, 5],
        'model__n_estimators': [100, 150, 200]
    }
}


def pipeline_grid_search_evaluation(model_name: str, grid_search_pipeline: GridSearchCV, X_test: pd.DataFrame,
                                    y_test) -> float:
    grid_search_pipeline_accuracy = grid_search_pipeline.best_estimator_.score(X_test, y_test)
    print("Accuracy: {}".format(grid_search_pipeline_accuracy))
    print("The best parameters are %s with a score of  %0.2f"
          % (grid_search_pipeline.best_params_, grid_search_pipeline.best_score_))

    grid_results = pd.concat([pd.DataFrame(grid_search_pipeline.cv_results_["params"]),
                              pd.DataFrame(grid_search_pipeline.cv_results_["mean_test_score"],
                                           columns=["Accuracy"])], axis=1)
    # Reshaping the data
    parameter_names = list(param_grid[model_name].keys())
    parameter_names = [parameter_name for parameter_name in parameter_names if parameter_name not in ['model']]
    if len(parameter_names) > 1:
        for parameter_name_1 in parameter_names:
            for parameter_name_2 in parameter_names:
                if parameter_name_1 != parameter_name_2:
                    print("parameter names for contour plotting: ", parameter_names)
                    grid_contour = grid_results.groupby(parameter_names).mean()
                    # Pivoting the estimators rows to columns
                    grid_contour = grid_contour.reset_index()
                    grid_contour = grid_contour[[parameter_name_1, parameter_name_2, 'Accuracy']]
                    grid_contour.columns = [parameter_name_1, parameter_name_2, 'Accuracy']

                    grid_pivot = grid_contour.pivot(parameter_name_1, parameter_name_2)
                    # x is the estimators, y is the features and z is the accuracy
                    x = grid_pivot.columns.levels[1].values
                    y = grid_pivot.index.values
                    z = grid_pivot.values

                    layout = go.Layout(
                        xaxis=go.layout.XAxis(
                            title=go.layout.xaxis.Title(
                                text=parameter_name_1)
                        ),
                        yaxis=go.layout.YAxis(
                            title=go.layout.yaxis.Title(
                                text=parameter_name_2)
                        ))
                    fig = go.Figure(data=[go.Contour(z=z, y=y, x=x)], layout=layout)
                    fig.update_layout(title='Hyperparameter tuning of parameters: {} and: {}'.format(parameter_name_1,
                                                                                                     parameter_name_2),
                                      autosize=False,
                                      width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
                    fig.show()

    return grid_search_pipeline_accuracy


def twitter_sentiment_classifier_training():
    # new: https://www.kaggle.com/kazanova/sentiment140
    # 0 negative, 1 positive
    data = pd.read_csv("sentimenttweets.csv", encoding='latin-1', usecols=['sentiment', 'text'])
    data = data.dropna()
    data['sentiment'].replace({4: 1}, inplace=True)
    show_sentiments(data)
    # Preprocess data
    data.text = data['text'].apply(preprocess_tweet_text)
    X_train, y_train, X_test, y_test = perform_train_test_split(data)
    print(X_train)
    print("Shape of Train data", X_train.shape)
    print("Shape of Test data", X_test.shape)
    tweets = data[data['sentiment'] == 0]
    print("Negative Tweets content word cloud")
    get_word_cloud(tweets)
    tweets = data[data['sentiment'] == 1]
    print("Positive Tweets content word cloud")
    get_word_cloud(tweets)
    # Perform transformations
    transformer = get_transformations()
    results = []
    names = []
    for model_name, model in models:
        pipeline = Pipeline(steps=[('transformer', transformer), ('model', model)])
        # Create StratifiedKFold object.
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        # choose the parameter grid using default and model-specific parameters
        print("available params for model name : {} are: {}".format(model_name, param_grid[model_name]))
        # Initiate the grid search
        grid_search_pipeline = GridSearchCV(pipeline, param_grid=param_grid[model_name], cv=skfold, n_jobs=-1,
                                            verbose=1,
                                            scoring=None)
        print("Performing grid search on model: {}".format(model_name))
        start_time = time.time()
        # Fit the grid search to the data
        print("nans in df: {}".format(X_train.isnull().sum().sum()))
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y_test))]
        grid_search_pipeline.fit(X_train, y_train)
        print("Training time: " + str((time.time() - start_time)) + ' ms')
        names.append(model_name)
        joblib.dump(grid_search_pipeline, f"models/{model_name}.pkl")
        print("****************GRID RESULTS***************")
        pipeline_result = pipeline_grid_search_evaluation(model_name=model_name,
                                                          grid_search_pipeline=grid_search_pipeline, X_test=X_test,
                                                          y_test=y_test)
        results.append(pipeline_result)
        print("****************TEST ACCURACY***************")
        print("Confusion Matrix", confusion_matrix(y_test, grid_search_pipeline.predict(X_test)))
        # plot confusion matrix
        visualize_plots.cf_matrix(y_test, grid_search_pipeline.predict(X_test))
        # classification report
        print(classification_report(y_test, grid_search_pipeline.predict(X_test)))
        # predict probabilities
        lr_probs = grid_search_pipeline.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # plot no skill roc curve
        visualize_plots.roc_curves(y_test, lr_probs)

        # Compare Algorithms
    plt.bar(x=names, height=results)
    plt.title('Algorithm Comparison')
    plt.ylabel('Model Accuracy')
    plt.xlabel('Model Name')
    plt.show()

    # Predict sentiments
    from tweet_sentiment_classifier_detection import tweet_sentiment_classifier_detection
    tweet_sentiment_classifier_detection(X_test)
