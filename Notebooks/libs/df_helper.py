import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

def vader_analyzer(df,date_col,text_col):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = [analyzer.polarity_scores(x) for x in df[text_col]]
    
    df_output = pd.DataFrame(sentiment)

    df[date_col] = pd.to_datetime(
    df[date_col],
    infer_datetime_format = True,
    utc = True    
    )
    df_output[date_col] = df[date_col].dt.date
    df_output['ticker'] = df['ticker']
    return df_output

def daily_sentiment(df,date_col,text_col):
    vader_df = vader_analyzer(df,date_col,text_col)
    vader_df = vader_df.groupby(['ticker',date_col])['ticker','pos','neg','neu','compound'].mean().reset_index()
    vader_df = vader_df[[date_col,'ticker','pos','neg','neu','compound']]
    return vader_df