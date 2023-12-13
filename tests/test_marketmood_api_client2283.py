import pytest
import pandas as pd
import os
import requests
import json
from textblob import TextBlob
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from marketmood_api_client2283 import classify_average_sentiment
from marketmood_api_client2283 import _fetch_financial_sentiment_data
from marketmood_api_client2283 import fetch_stock_prices  


# +
import pytest
import pandas as pd

def test_fetch_financial_sentiment_data_simple():
    company_name = "Tesla"
    
    # Call the function
    df = _fetch_financial_sentiment_data(company_name)

    # Check if the result is a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check if the DataFrame contains expected columns
    expected_columns = {'title', 'description', 'published_at', 'sentiment_score', 'sentiment_category', 'url'}
    assert set(df.columns).issuperset(expected_columns)

    # Check if the DataFrame is not empty
    assert not df.empty



# +
import pytest
import pandas as pd

def test_classify_average_sentiment():
    # Create a sample DataFrame with predefined sentiment scores
    data = {
        'sentiment_score': [-0.6, -0.2, 0.0, 0.3, 0.6]
    }
    df = pd.DataFrame(data)

    # Call the function
    average_score, sentiment_level = classify_average_sentiment(df)

    # Check the average score and sentiment level
    assert average_score == pytest.approx(0.02, 0.001)
    assert sentiment_level == "Neutral"

def test_classify_average_sentiment_empty_df():
    # Test with an empty DataFrame
    df_empty = pd.DataFrame()
    average_score, sentiment_level = classify_average_sentiment(df_empty)

    # Check the return values for an empty DataFrame
    assert average_score is None
    assert sentiment_level == "No Data"



# +
import pytest
import pandas as pd

def test_fetch_stock_prices():
    symbol = "AAPL"  # Apple Inc. as a commonly known stock symbol

    # Call the function
    df = fetch_stock_prices(symbol, period='5d')  # Fetch data for the last 5 days

    # Check if the result is a DataFrame
    assert isinstance(df, pd.DataFrame), "The result should be a pandas DataFrame"

    # Check if the DataFrame contains the expected columns
    expected_columns = ['Date', 'Close']
    assert set(df.columns) == set(expected_columns), "DataFrame should have Date and Close columns"

    # Check if the DataFrame is not empty
    assert not df.empty, "The DataFrame should not be empty"

    # Optionally, check the data types of the columns
    assert pd.api.types.is_datetime64_any_dtype(df['Date']), "Date column should be datetime type"
    assert pd.api.types.is_float_dtype(df['Close']), "Close column should be float type"

# -

# !poetry run pytest


