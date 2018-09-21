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
        if i % 500 == 0:      
            print ('\nWe have gathered:',len(tweets_data), 'tweets.\n')
            print('This is one of them:', tweet['text'])
        if i >= 20000: # Define size of subset
            del i
            del line
            del tweet
            break
    except Exception as e:
        print (e)
        continue

for i in range(len(tweets_data)):
    print(tweets_data[i]['lang'])

list(map(lambda tweet: tweet['text'], tweets_data))

tweets = pd.DataFrame()

tweets['text'] =    list(map(lambda tweet: tweet['text'], tweets_data))
tweets['lang'] =    list(map(lambda tweet: tweet['lang'], tweets_data))
tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))
tweets['location'] = list(map(lambda tweet: tweet['place']['bounding_box'] if tweet['place']['bounding_box'] != None else None, tweets_data))

# See this website for more info on the metadata: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/geo-objects.html



# A function that extracts the hyperlinks from the tweet's content.
def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''

# A function that checks whether a word is included in the tweet's content
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

tweets['link'] = tweets['text'].apply(lambda tweet: extract_link(tweet))
print(tweets['link'])


# Quote someone and see if electoral result is correlated with sentiment
    # Maybe use a second source for confirmation... 
    # Sentiment key-word correlated with voting
    # 
# Quote paper on twitter and pol. polarization and bring attention to the 10.000 trolls









