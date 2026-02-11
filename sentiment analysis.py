import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy

# >>> STEP 1: Configuration & API Setup <<<

NEWS_API_KEY = "YOUR_NEWS_API_KEY"
TWITTER_BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"
SEARCH_TERM = "stock market OR finance OR S&P500"  # customize as needed

# Set up Twitter API client (Tweepy)
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# >>> STEP 2: Fetch Data <<<

def fetch_news(query, api_key, pages=1):
    articles = []
    for page in range(1, pages+1):
        url = f"https://newsapi.org/v2/everything?q={query}&pageSize=50&page={page}&apiKey={api_key}"
        resp = requests.get(url).json()
        if "articles" in resp:
            for art in resp["articles"]:
                articles.append(art["title"] or "")
        else:
            break
    return articles

def fetch_tweets(query, max_tweets=100):
    tweets = []
    response = client.search_recent_tweets(query=query, max_results=min(max_tweets, 100), tweet_fields=["text", "lang"])
    for tweet in response.data or []:
        if tweet.lang == "en":
            tweets.append(tweet.text)
    return tweets

# >>> STEP 3: Preprocessing <<<

def preprocess(text):
    # Basic text normalization
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text

# >>> STEP 4: Sentiment Analysis <<<

def analyze_sentiment(texts):
    sentiments = []
    for text in texts:
        score = analyzer.polarity_scores(text)["compound"]
        sentiments.append(score)
    return sentiments

# >>> STEP 5: Aggregate Results <<<

def summarize(sentiments):
    if not sentiments:
        return "No data"
    avg_sentiment = sum(sentiments)/len(sentiments)
    if avg_sentiment > 0.05:
        return f"Overall Sentiemnt: Positive ({avg_sentiment:.2f})"
    elif avg_sentiment < -0.05:
        return f"Overall Sentiment: Negative ({avg_sentiment:.2f})"
    else:
        return f"Overall Sentiment: Neutral ({avg_sentiment:.2f})"

# >>> RUN ANALYSIS <<<

if __name__ == "__main__":
    print("Fetching news...")
    news_titles = fetch_news(SEARCH_TERM, NEWS_API_KEY, pages=2)
    print(f"Fetched {len(news_titles)} news headlines.")

    print("Fetching tweets...")
    tweets = fetch_tweets(SEARCH_TERM, max_tweets=100)
    print(f"Fetched {len(tweets)} tweets.")

    all_texts = news_titles + tweets
    preprocessed = [preprocess(t) for t in all_texts]
    sentiments = analyze_sentiment(preprocessed)
    summary = summarize(sentiments)
    print(summary)