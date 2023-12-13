import pandas as pd
import os
import requests
import json
from textblob import TextBlob
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _fetch_financial_sentiment_data(company_name):
    """
    Fetches financial market data for a specific company and analyzes sentiment of the articles using VADER.
    Retrieves the last 100 news items about the company, including the news time and URL.
    :param company_name: Name of the company.
    :return: DataFrame with article details, sentiment scores, publication times, and URLs.
    """
    api_endpoint = 'https://newsapi.org/v2/everything'
    api_key = "f1d6a48ce4574b4ba30eac6e43162957"  
    params = {
        'q': company_name,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100  # Fetch up to 100 articles
    }

    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return pd.DataFrame()
    except Exception as err:
        print(f'Other error occurred: {err}')
        return pd.DataFrame()

    articles = response.json().get('articles', [])
    data = []
    analyzer = SentimentIntensityAnalyzer()

    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        published_at = datetime.strptime(article.get('publishedAt', '')[:10], '%Y-%m-%d').date() if article.get('publishedAt') else None
        url = article.get('url', '')

        text = f"{title} {description}".strip()
        sentiment_score = analyzer.polarity_scores(text)['compound']
        sentiment_category = 'Positive' if sentiment_score > 0.05 else 'Negative' if sentiment_score < -0.05 else 'Neutral'

        data.append({
            'title': title,
            'description': description,
            'published_at': published_at,
            'sentiment_score': sentiment_score,
            'sentiment_category': sentiment_category,
            'url': url
        })

    df = pd.DataFrame(data)
    return df


# Example usage
company_name = "Tesla"
df = _fetch_financial_sentiment_data(company_name)
df.head()


def classify_average_sentiment(df):
    """
    Computes the average sentiment score and classifies it into levels.
    :param df: DataFrame with sentiment scores.
    :return: A tuple containing the average sentiment score and its classification.
    """
    if df.empty:
        return (None, "No Data")

    average_score = df['sentiment_score'].mean()

    if average_score <= -0.5:
        sentiment_level = "Strongly Negative"
    elif -0.5 < average_score <= -0.1:
        sentiment_level = "Negative"
    elif -0.1 < average_score < 0.1:
        sentiment_level = "Neutral"
    elif 0.1 <= average_score < 0.5:
        sentiment_level = "Positive"
    else:
        sentiment_level = "Strongly Positive"

    return (average_score, sentiment_level)


# +
# Fetch sentiment data
company_name = "Apple"
sentiment_data = _fetch_financial_sentiment_data(company_name)

# Classify average sentiment
average_sentiment, sentiment_level = classify_average_sentiment(sentiment_data)

print(f"Average Sentiment Score for {company_name}: {average_sentiment}")
print(f"Sentiment Level: {sentiment_level}")

# +
# Get the company's stock price and see the correlation between the sentiment scores and the stock price

# +
import yfinance as yf

def fetch_stock_prices(symbol, period='1mo'):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    return df[['Date', 'Close']]


# -

def align_data(df_sentiment, df_stock_prices):
    df_sentiment['date'] = pd.to_datetime(df_sentiment['published_at']).dt.date
    df_stock_prices['date'] = pd.to_datetime(df_stock_prices['Date']).dt.date
    df_combined = pd.merge(df_sentiment, df_stock_prices, left_on='date', right_on='date')
    return df_combined


def analyze_correlation(df_combined):
    return df_combined[['sentiment_score', 'Close']].corr()



# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_data(df_combined):
    plt.figure(figsize=(12, 6))

    # Ensure data is in correct format for Matplotlib
    dates = df_combined['date'].values
    sentiment_scores = df_combined['sentiment_score'].values
    stock_prices = df_combined['Close'].values

    # Create a plot with two y-axes
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Score', color=color)
    ax1.plot(dates, sentiment_scores, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Stock Price', color=color)
    ax2.plot(dates, stock_prices, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Sentiment Score and Stock Price Over Time')
    fig.tight_layout()  
    plt.show()



# -

# # Example Usage

# +
# Example usage
company_name = "TSLA"  # Tesla
stock_symbol = "TSLA"

# Fetch Data
df_stock_prices = fetch_stock_prices(stock_symbol)
df_sentiment = _fetch_financial_sentiment_data(company_name) 

# Align and Combine Data
df_combined = align_data(df_sentiment, df_stock_prices)

# Analyze Correlation
correlation = analyze_correlation(df_combined)
print(correlation)

# Plot Data 
plot_data(df_combined)

# -






