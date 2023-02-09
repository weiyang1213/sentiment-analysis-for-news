# Reference: https://github.com/ARGULASAISURAJ/Stock-Price-visualisation-Web-App/blob/master/Visualise_Stock_market_Prices_Google_App.py
# Reference: host to heroku https://medium.com/analytics-vidhya/how-to-deploy-a-streamlit-app-with-heroku-5f76a809ec2e

# LIbrary We Use for modal traning algorithm , web scarping, show graph and preprocessing
import yfinance as yf
import streamlit as st
import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from pandas import DataFrame
import re
import string
import time
import json


#---------------------------------------------------------------------------------------#

st.set_page_config(layout="wide")

st.write('# Sentiment Analysis With CNBC News Real Time Data')
st.write('###  Enjoy!')

#---------------------  Function Sentiment Analysis for CNBC stock news  --------------#

title_list = []
time_list = []

# Try to scrap the lastest news block with the headlines
cnbc_news = requests.get('https://www.cnbc.com/world/?region=world').text
soup = BeautifulSoup(cnbc_news, 'lxml')
stock_news_block = soup.find('div', class_="undefined LatestNews-isHomePage LatestNews-isIntlHomepage")
stock_news_list = stock_news_block.find('ul',class_="LatestNews-list").text
headline = stock_news_block.find_all('div', attrs={'class':"LatestNews-headlineWrapper"})

title = stock_news_block.find_all('a', attrs= {'class':"LatestNews-headline"})
lenght = len(title)

for x in title:
    # store in variable
    titles = x.text
    #print(titles)
    title_list.append(titles)

# get time from CNBC lastest news block 
cnbc_news = requests.get('https://www.cnbc.com/world/?region=world').text
soup = BeautifulSoup(cnbc_news, 'lxml')
stock_news_block = soup.find('div', class_="undefined LatestNews-isHomePage LatestNews-isIntlHomepage")
time = stock_news_block.find_all('time', attrs={'class':"LatestNews-timestamp"})
lenght = len(time)

#put news time into a list using for loop
for y in time:
    time = y.text
    time_list.append(time)

# Store data we scrap into a list
data = {
        "Stock Title": title_list,
        "Time": time_list,

    }
df = DataFrame(data, columns=[
        "Stock Title" , "Time"
       
    ])

# preprocessing
df = df.dropna()
df['Stock Title after preprocessing'] = df['Stock Title'].str.lower()
df['Stock Title after preprocessing'] = df['Stock Title after preprocessing'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
#remove stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])
df['Stock Title after preprocessing']= df['Stock Title after preprocessing'].apply(lambda x: remove_stopwords(x))

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
df['Stock Title after preprocessing'] = df['Stock Title after preprocessing'].apply(lambda text: lemmatize_words(text))

# -------------- Sentiment Analysis Apply Coding Part -----#

from nltk.sentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

df['scores'] = df['Stock Title after preprocessing'].apply(lambda review: vader.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['sentiment'] = df['compound'].apply(lambda c:'positive' if c>=0 else 'negative')

@st.cache
def convert_df(df):
     return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.write(' ')
if st.checkbox('Show Current CNBC News Title and Sentiment Analysis result to analysis what news will affect stock price'):
    df

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='sentiment_analysis_CBNC_Stock_News.csv',
        mime='text/csv',
    )

