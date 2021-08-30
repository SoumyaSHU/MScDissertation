import joblib
import pandas as pd


def tweet_sentiment_classifier_detection(tweets):
    pipeline = joblib.load('models/logistic_regression.pkl')
    transformed_data = pipeline.predict_proba(tweets)
    transformed_data = transformed_data[:, 1]
    tweets = pd.DataFrame(tweets)
    tweets['score'] = transformed_data
    print(tweets)
    print(tweets['score'])
    # selecting positive tweets
    ptweets = tweets.loc[tweets['score'] > 0.4]
    print('\nResult dataframe :\n', ptweets)
    # printing first 5 positive tweets
    print("\n\nPositive tweets:")
    print(ptweets[['text']].head(5))
    print('\nResult dataframe :\n', len(ptweets))
    print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
    # selecting negative tweets
    ntweets = tweets.loc[tweets['score'] < 0.4]
    print('\nResult dataframe :\n', len(ntweets))
    print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
    # printing first 5 negative tweets
    print("\n\nNegative tweets:")
    print(ntweets[['text']].head(5))
