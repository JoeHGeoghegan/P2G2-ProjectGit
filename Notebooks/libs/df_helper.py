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

def date_filling_concat(df1:pd.DataFrame,df1_date_col,df2:pd.DataFrame,df2_date_col):
    """
    The df that needs to be stretched needs to have its date col setup to be in the "YYYY-MM-DD-YYYY-MM-DD" format with the start time on left and end time on right
    """
    if len(df1) > len(df2):
        into_df = df1
        into_date_col = df1_date_col
        stretch_df = df2
        stretch_date_range_col = df2_date_col
    elif len(df1) < len(df2):
        stretch_df = df1
        stretch_date_range_col = df1_date_col
        into_df = df2
        into_date_col = df2_date_col
    else: # No need to fill if they are equal, just concat, what are you even doing here?
        return pd.concat([df1,df2],axis='columns',join='inner')
    