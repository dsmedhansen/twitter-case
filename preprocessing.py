#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:16:32 2018

@author: Daniel
"""
import json
import os
import pandas as pd
import re

#%%

PATH = '/Users/Daniel/Google Drive/Master/Fundamentals of data science/'
os.chdir(PATH)
print("Working directory: %s" % os.getcwd() )

tweets_collection = "geotagged_tweets.jsons"
tweets_data = []
tweets_file = open(tweets_collection, "r")

i = 0

for line in tweets_file:

    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
        i += 1  
        if i % 100 == 0:      
            print ('\nWe have gathered:',len(tweets_data), 'tweets.\n')
            print('This is one of them:', tweet['text'])
        if i >= 10000: # Define size of subset
            del i, line, tweet, tweets_collection
            break
    except Exception as e:
        print (e)
        continue

#for i in range(len(tweets_data)):
    #print(tweets_data[i]['lang'])

list(map(lambda tweet: tweet['text'], tweets_data))

tweets = pd.DataFrame()

#%%

tweets['text'] =    list(map(lambda tweet: tweet['text'], tweets_data))
tweets['lang'] =    list(map(lambda tweet: tweet['lang'], tweets_data))
tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))

#%%
tweets['location'] = list(map(lambda tweet: tweet['place']['bounding_box'] if tweet['place']['bounding_box'] != None else None, tweets_data))
    # This is not working if I use the whole dataset...? 
# See this website for more info on the metadata: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/geo-objects.html

#%%

# A function that extracts the hyperlinks from the tweet's content.
def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''

# A function that checks whether a word is included in the tweet's content
def word_in_text(word, text): # Redundant?
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

#%%

tweets['link'] = tweets['text'].apply(lambda tweet: extract_link(tweet))
print(tweets['link'])

#%%

def remove_link(text): # Remove links from text as first step in cleaning of data
    for link in text:
        result = re.sub(r"http\S+", "", text)
        print ("\n\nLink free:\n",result)
        return result
    
tweets['text'] = tweets['text'].apply(lambda tweet: remove_link(tweet)) # Remove links from text

#%%
# Move all mentions to a separate column

def twitter_handle(tweet):
    handle = r'@[a-zA-Z_0-9]{4,}' # Regex for Twitter handle
    match = re.findall(handle, tweet)
    if match:
        return match
    return '' 

tweets['handles'] = tweets['text'].apply(lambda tweet: twitter_handle(tweet)) # Move handles to column

#%%

def remove_handle(text): # Remove handles from text
    for link in text:
        result = re.sub(r'@[a-zA-Z_0-9]{4,}', '', text)
        print ("\n\nHandle free:\n", result)
        return result
    
tweets['text'] = tweets['text'].apply(lambda tweet: remove_handle(tweet)) # Remove all twitter-handles from text 

#%% A bit more cleaning and we'll be ready for the sentiment analysis

# Remove special characters and numbers.... 

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))

# Create a set of punctuation words 
exclude = set(string.punctuation) 

# This is the function makeing the lemmatization
lemma = WordNetLemmatizer()

# In this function we perform the entire cleaning
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

#%%
# This is the clean corpus.
    # This part is not working anymore: 'NoneType' object has no attribute 'lower'
    # I think the problem occurs when I remove twitter handles and hyperlinks
    # because this means that some cells are NA's, which returns an empty object

tweets['text_clean'] = [clean(doc).split() for doc in tweets['text']] # Check df: ISIS is lemmatized to isi which might be problem

# change "isi" to isis (this doesnt matter for the sentiment analysis but it will for the topic modelling)

#%%

from textblob import TextBlob # Textblob for sentiment analysis

tweets['text_clean'] = tweets['text_clean'].apply(lambda tweet: ' '.join(tweet))
tweets['blob_sentiment'] = tweets['text_clean'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Run two different sentiment analysis and correlate the outcomes..
# Find model that includes emojis

#%%
from nltk.sentiment import vader
import seaborn as sns

senti= vader.SentimentIntensityAnalyzer()

tweets['vader_sentiment'] = tweets['text_clean'].apply(lambda tweet: senti.polarity_scores(tweet)['compound'])

# Impliment SentiStrength (better for short texts) 
# Website: http://sentistrength.wlv.ac.uk/

#%%

corrmatrix = tweets[['blob_sentiment', 'vader_sentiment']].corr()
corrmatrix

sns.heatmap(corrmatrix) # Correlations below 0.6... 

#%%

from statsmodels.stats.weightstats import ttest_ind

blob = []   
vader = []
    
for score in tweets['blob_sentiment']:
    blob.append(score)
    
for score in tweets['vader_sentiment']:
    vader.append(score)

results_sentiments = ttest_ind(blob, vader)

print('t({2:.0f}) = {0:.3f}, p = {1:.3F}'.format(*results_sentiments))

# Use model trained on bigger corpus or stick with vader/blob?
# Do we need to train a better model for sentiment analysis or should we stick with Vader/Blob?

#%%
import gensim
from gensim import corpora

doc_clean = tweets['text_clean'].tolist()

#%%

# Delete DF to free up memory before running LDA

#%%

dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=100)

# Print 2 topics and describe then with 4 words.
topics = ldamodel.print_topics(num_topics=10, num_words=10)

i=0
for topic in topics:
    print ("Topic",i ,"->", topic)     
    i+=1
    
# Maybe we should just remove the handles since all topics now contain 
# words such as @realdonaltrump
