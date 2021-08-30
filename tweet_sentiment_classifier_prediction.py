import csv
import joblib
import pandas as pd
import re
import tweepy
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string
from datetime import date, datetime
import matplotlib.dates as mdates
import seaborn as sns


# how to run flask application
# set FLASK_APP=tweet_sentiment_classifier_prediction.py
# set FLASK_DEBUG=1
# $env:FLASK_APP = "tweet_sentiment_classifier_prediction.py"
# python -m  flask run
# or
# flask run


def get_positive_word_cloud(tweets):
    allwords = ''.join([twts for twts in tweets])
    wordcloud = WordCloud(width=500, height=300,
                          random_state=21,
                          background_color='white',
                          max_font_size=119).generate(allwords)
    # plot the WordCloud image
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file("static/positive_wordcloud.png")
    plt.show()


def get_negative_word_cloud(tweets):
    allwords = ''.join([twts for twts in tweets])
    wordcloud = WordCloud(width=500, height=300,
                          random_state=21,
                          background_color='white',
                          max_font_size=119).generate(allwords)
    # plot the WordCloud image
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file("static/negative_wordcloud.png")
    plt.show()


def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove '#' from tweet
    tweet = re.sub(r'\#\w+|\#', '', tweet)
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


# start flask
from flask import Flask, render_template
import os

IMAGE_FOLDER = os.path.join('static')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


# render deafult webpage
@app.route('/')
def tweet_sentiment_classifier_prediction():
    # Twitter API credentials
    access_token = "1310941637024141313-GjXOWqYceSJLzo2MBEippBXMQEgNwj"
    access_token_secret = "Q1ubUCKc8zGzDXOtumKh9gVPBn75VaDBuI0o36EkcGbFJ"
    consumer_key = "RxiS9SqRWzsq479r28UKtKsBJ"
    consumer_secret = "hB0Wr7K9iClKPFb6JDQq9lEMbdiE9Ohwedw41JHRVws8AFo4n3"
    # attempt authentication
    authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # set access token and secret
    authenticate.set_access_token(access_token, access_token_secret)
    # create tweepy API object to fetch tweets
    api = tweepy.API(authenticate, wait_on_rate_limit=True)
    csvFile = open('LiveTweets.csv', 'w', newline='', encoding="utf8")
    csvWriter = csv.writer(csvFile)
    header = ['timestamp', 'text']
    csvWriter.writerow(header)
    search_words = "COVID19, CoronavirusPandemic"  # enter your words
    new_search = search_words + " -filter:retweets"
    for tweet in tweepy.Cursor(api.search, q=new_search, count=100,
                               lang="en",
                               since_id=1).items():
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

    data = pd.read_csv("LiveTweets.csv", usecols=['text'])
    data.text = data['text'].apply(preprocess_tweet_text)
    Subjectivity = data['text'].apply(getSubjectivity)
    Polarity = data['text'].apply(getPolarity)
    Analysis = Polarity.apply(getAnalysis)
    # Plot the Polarity and Subjectivity
    fig = plt.figure(figsize=(8, 6))
    for i in range(0, data.shape[0]):
        plt.scatter(Polarity, Subjectivity, color='Blue')
    plt.title('Sentiment Analysis')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.savefig('static/polsubj.png', dpi=fig.dpi)
    pipeline = joblib.load('models/logistic_regression.pkl')
    transformed_data = pipeline.predict_proba(data)
    transformed_data = transformed_data[:, 1]
    data = pd.read_csv("LiveTweets.csv")
    data.text = data['text'].apply(preprocess_tweet_text)
    sentiment = pd.DataFrame(data)
    sentiment['score'] = transformed_data
    print(sentiment)

    # selecting positive tweets
    ptweets = sentiment.loc[sentiment['score'] > 0.5]
    print('\nResult dataframe :\n', ptweets)
    get_positive_word_cloud(ptweets['text'])
    positive_tweets = ptweets[['text']].head(5)
    print('\nResult dataframe :\n', len(ptweets))
    print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(sentiment)))
    # selecting negative tweets
    ntweets = sentiment.loc[sentiment['score'] < 0.5]
    get_negative_word_cloud(ntweets['text'])
    print('\nResult dataframe :\n', len(ntweets))
    print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(sentiment)))
    negative_tweets = ntweets[['text']].head(5)
    print(negative_tweets)
    total_tweets = len(sentiment)
    total_positive = len(ptweets)
    total_negative = len(ntweets)
    # extracting date from timestamp
    sentiment['date'] = pd.to_datetime(sentiment['timestamp'], format='%Y-%m-%d')
    sentiment['date'] = sentiment['date'].dt.strftime('%Y-%m-%d')
    sns.lineplot(x=sentiment['date'], y=sentiment['score'], data=sentiment)
    plt.xticks(rotation=45)
    plt.title('Sentiment Timeline')
    plt.tight_layout()
    plt.savefig('static/timeline.png')
    plt.show()

    slices_tweets = [format(100 * len(ptweets) / len(sentiment)), format(100 * len(ntweets) / len(sentiment))]
    analysis = ['Positive', 'Negative']
    colors = ['g', 'r']
    plt.pie(slices_tweets, labels=analysis, startangle=-40, autopct='%.1f%%')  # to generate the pie chart
    plt.savefig('static/piechart.png')  # to save the local copy of the piechart in your PC
    plt.show()  # to display the generated chart
    now = datetime.now()
    pos = os.path.join(app.config['UPLOAD_FOLDER'], 'positive_wordcloud.png')
    neg = os.path.join(app.config['UPLOAD_FOLDER'], 'negative_wordcloud.png')
    timeline = os.path.join(app.config['UPLOAD_FOLDER'], 'timeline.png')
    pie_chart = os.path.join(app.config['UPLOAD_FOLDER'], 'piechart.png')
    happy = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.png')
    sad = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return render_template('home.html', datetime=dt_string, pos_wordcloud=pos,
                           neg_wordcloud=neg,
                           sentiment_timeline=timeline,
                           piechart=pie_chart,
                           pos_tweets=positive_tweets,
                           neg_tweets=negative_tweets,
                           total_tweets=total_tweets,
                           total_positive=total_positive,
                           total_negative=total_negative,
                           happy=happy,
                           sad=sad)

