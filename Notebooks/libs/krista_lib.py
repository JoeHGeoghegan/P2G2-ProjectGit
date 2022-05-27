#Imports
import os
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("NEWSAPI_KEY")

newsapi = NewsApiClient(api_key=api_key)

#Get news articles on certain topic based on keywords
def get_news(keywords):  
    news_article = newsapi.get_everything(
            q = keywords, language='en', sort_by= 'relevancy'
    )
    return news_article

#Creates dataframe of the articles chosen 
def form_df(keywords):
    news = get_news(keywords)['articles']

    articles = []
    for article in news:
        try:
            title = article['title']
            description = article['description']
            text = article['content']
            date = article['publishedAt'][:10]

            articles.append({
                'title' : title,
                'description' : description,
                'text' : text,
                'date' : date,
                'language' : 'en'
            })
        except AttributeError:
            pass
    
    return pd.DataFrame(articles)