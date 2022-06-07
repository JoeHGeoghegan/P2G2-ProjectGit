#Imports
import os
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
load_dotenv()
from collections import Counter
from nltk.corpus import reuters, stopwords
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('reuters')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

api_key = os.getenv("NEWSAPI_KEY")

newsapi = NewsApiClient(api_key=api_key)

#Get news articles on certain topic based on keywords
def get_news(keywords):  
    news_article = newsapi.get_everything(
            q = keywords, language='en', sort_by= 'relevancy', from = "2012-06-01", to = '2022-06-01'
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


#Code that counts number of articles present about a topic
def news_density(keywords):
    count = 0 
    for dict in get_news(keywords)['articles']:
        count += 1
    return count

#Tokenizer function
def tokenizer(text):
    """Tokenizes text."""
    
    # Remove the punctuation from text
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', text)

    # Create a tokenized list of the words
    words = word_tokenize(re_clean)
    
    # Lemmatize words into root words
    lemmatizer = WordNetLemmatizer()
    lem = [lemmatizer.lemmatize(word) for word in words]

    # Remove the stop words
    sw = set(stopwords.words('english'))
    
    # Convert the words to lowercase
    tokens = [word.lower() for word in lem if word.lower() not in sw]
    
    
    return tokens

#Takes text column and turn into list of words to iterate and analyze
def text_splitter(df):
    df["text splitted"] = df.text.str.lower().str.replace('[^\w\s]','').str.split()
    df["text splitted"].transform(lambda x: Counter(x)).sum()
    return df

#Counts occurence of certain words in text to see how much a certain stock is talked about
def word_occurence(df, word):
    df = text_splitter(df)
    count = 0 
    for rows in df['text splitted']:
        for words in rows:
            if words == word:
                count += 1
    return count