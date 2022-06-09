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

def keyword_filter_2(df, keywords):

    filtered_list = []
    for keyword in keywords:
        for text in df['text']:
            if keyword in str(text):
                filtered_list.append(text)

    filtered_df = pd.concat([df['datetime'], pd.DataFrame(filtered_list).rename(columns = {0: 'text'})], axis = 1).dropna()
    return filtered_df

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

def daily_mean_2(source, df):

    daily_mean_df = df.groupby(['datetime'])[[f'{source}_compound_sentiment', f'{source}_positive_sentiment', f'{source}_neutral_sentiment', f'{source}_negative_sentiment']].mean().reset_index()

    return daily_mean_df
