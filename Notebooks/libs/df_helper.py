import SentimentIntensityAnalyzer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

def vader_analyzer(df,text_col):
    analyzer = SentimentIntensityAnalyzer()
    df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df[text_col]]
    df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df[text_col]]
    df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df[text_col]]
    df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df[text_col]]

    df['date'] = pd.to_datetime(
    df['date'],
    infer_datetime_format = True,
    utc = True    
    )
    df['date'] = df['date'].dt.date
    
    return df

def daily_sentiment(df,text_col):
    vader_df = vader_analyzer(df,text_col)
    vader_df = vader_df.groupby(['ticker','date'])['ticker','pos','neg','neu','compound'].mean().reset_index()
    vader_df = vader_df[['date','ticker','pos','neg','neu','compound']]
    return vader_df