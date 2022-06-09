import pandas as pd
import datetime as dt
import pmaw
import newsapi
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

def articles_pull(keywords):

    newsapi = NewsApiClient(api_key = newsapi_key)
    newsapi_response = newsapi.get_everything(q = keywords, language = 'en', sort_by = 'publishedAt')['articles']
    articles_list = []
    for article in newsapi_response:
        try:
            title = article['title']
            description = article['description']
            text = article['content']
            date = article['publishedAt']
            articles_list.append({'date' : date, 'text' : text})
            articles = pd.DataFrame(articles_list).rename(columns = {'date': 'datetime'}).sort_values('datetime')
            articles['datetime'] = pd.to_datetime(articles['datetime'], infer_datetime_format = True, errors = 'coerce')
        except AttributeError:
            pass
    
    return articles

def subreddit_pull(subreddit, limit, after, before):

    pushshift = pmaw.PushshiftAPI()
    comments_response = pushshift.search_comments(subreddit = subreddit, limit = limit, after = after, before = before)
    comments_original = pd.DataFrame(comments_response)
    comments_original['datetime'] = comments_original.apply(lambda row : dt.datetime.fromtimestamp(row['created_utc']), axis = 1)
    comments_original['datetime'] = pd.to_datetime(comments_original['datetime'])
    comments = comments_original[['datetime', 'body']].rename(columns = {'body': 'text'}).set_index('datetime')
    
    return comments

def keyword_filter(df, keywords):

    filter = df['text'].str.contains(keywords[0])
    for keyword in keywords[1:]:
        filter = filter | df['text'].str.contains(keyword)

    return df[filter]

def articles_vader_analyzer(df):
    
    analyzer = SentimentIntensityAnalyzer()
    df['articles_compound_sentiment'] = [analyzer.polarity_scores(x)['compound'] for x in df['text']]
    df['articles_positive_sentiment'] = [analyzer.polarity_scores(x)['pos'] for x in df['text']]
    df['articles_neutral_sentiment'] = [analyzer.polarity_scores(x)['neu'] for x in df['text']]
    df['articles_negative_sentiment'] = [analyzer.polarity_scores(x)['neg'] for x in df['text']]

def reddit_vader_analyzer(subreddit, df):

    analyzer = SentimentIntensityAnalyzer()
    df[f'{subreddit}_compound_sentiment'] = [analyzer.polarity_scores(x)['compound'] for x in df['text']]
    df[f'{subreddit}_positive_sentiment'] = [analyzer.polarity_scores(x)['pos'] for x in df['text']]
    df[f'{subreddit}_neutral_sentiment'] = [analyzer.polarity_scores(x)['neu'] for x in df['text']]
    df[f'{subreddit}_negative_sentiment'] = [analyzer.polarity_scores(x)['neg'] for x in df['text']]

    return df

def daily_mean(df):

    df['datetime'] = pd.to_datetime(df['datetime'])
    daily_mean_df = df.set_index('datetime').groupby(pd.Grouper(freq = 'd')).mean().reset_index()

    return daily_mean_df

def process():

    stock_data = pd.read_csv('./Data/Cleaned_Data/stock_data.csv', parse_dates = True, infer_datetime_format = True)
    
    stockmarket_comments = pd.read_csv('../../sandbox/stockmarket_comments_large.csv', lineterminator = '\n', parse_dates = True, infer_datetime_format = True)

    stock_data['date'] = pd.to_datetime(stock_data['date'], infer_datetime_format = True, errors = 'coerce')
    stock_data = stock_data.set_index('date')
    stock_data.index.name = None

    keywords = {
    'NFLX': ['NFLX', 'nflx', 'Netflix', 'netflix'],
    'FB': ['FB', 'fb', 'Facebook', 'facebook'],
    'UBER': ['UBER', 'uber', 'Uber'],
    'MCHP': ['MCHP', 'mchp', 'Microchip Technology'],
    'ABNB': ['ABNB', 'abnb', 'AirBnB', 'airbnb'],
    'FANG': ['FANG', 'fang', 'Diamondback Energy', 'diamondback energy', 'Diamondback', 'diamondback'],
    'MRO': ['MRO', 'mro', 'Marathon Oil', 'marathon oil'],
    'DVN': ['DVN', 'dvn', 'Devon Energy', 'devon energy'],
    'SPWR': ['SPWR', 'spwr', 'SunPower', 'Sunpower', 'sunpower'],
    'REGI': ['REGI', 'regi', 'Renewable Energy Group', 'renewable energy group'],
    'MTRX': ['MTRX', 'mtrx', 'McKinsey & Company', 'McKinsey & Co', 'Mckinsey & Co', 'McKinsey', 'Mckinsey', 'mckinsey'],
    'BLK': ['BLK', 'blk', 'BlackRock', 'Blackrock', 'blackrock'],
    'PYPL': ['PYPL', 'pypl', 'PayPal', 'Paypal', 'paypal'],
    'MELI': ['MELI', 'meli', 'MercadoLibre', 'Mercadolibre', 'mercadolibre'],
    'SOFI': ['SOFI', 'sofi', 'SoFi', 'Sofi']}

    for asset in keywords:

        asset_prices_volume = stock_data[stock_data['ticker'] == asset].drop(columns = 'ticker')

        asset_stockmarket_comments = keyword_filter(stockmarket_comments.fillna(''), keywords[asset])

        asset_stockmarket_daily_sentiment = daily_mean(reddit_vader_analyzer('stockmarket', asset_stockmarket_comments))
        asset_stockmarket_daily_sentiment['datetime'] = pd.to_datetime(asset_stockmarket_daily_sentiment['datetime'], infer_datetime_format = True, errors = 'coerce')
        asset_stockmarket_daily_sentiment = asset_stockmarket_daily_sentiment.set_index('datetime')
        asset_stockmarket_daily_sentiment.index = asset_stockmarket_daily_sentiment.index.date

        asset_df = pd.concat([asset_stockmarket_daily_sentiment, asset_prices_volume], axis = 1)
        asset_df.ffill(axis = 0, inplace = True)
        asset_df.dropna(inplace = True)
        asset_df.to_csv(f'./Data/Cleaned_Data/{asset}.csv')

    return None