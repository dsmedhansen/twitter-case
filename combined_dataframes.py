# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.sentiment import vader
#import seaborn as sns

file = open('/home/martijn/Downloads/geotagged_tweets_20160812-0912.jsons', 'r')
#file = open('/home/martijn/Downloads/tweets_n100000.json', 'r')
final_file = open('/home/martijn/Downloads/final_frame_full.txt', 'w')

tweets_data = []

for line in file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
#        print ("Imported tweet created at:", tweet['created_at'])
#        print ("Tweet content: \n", tweet['text'], "\n")
    except Exception as e:
        print (e)
        continue

tweets = pd.DataFrame()
tweets['text'] =    list(map(lambda tweet: tweet['text'], tweets_data))
tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))
file.close()

def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''

tweets['link'] = tweets['text'].apply(lambda tweet: extract_link(tweet))

def remove_link(text): # Remove links from text as first step in cleaning of data
    for link in text:
        result = re.sub(r"http\S+", "", text)
        if result:
            return result
        return ''
    
tweets['text'] = tweets['text'].apply(lambda tweet: remove_link(tweet))

def twitter_handle(tweet):
    handle = r'@[a-zA-Z_0-9]{4,}' # Regex for Twitter handle
    match = re.findall(handle, tweet)
    if match:
        return match
    return ''

tweets['handles'] = tweets['text'].apply(lambda tweet: twitter_handle(tweet))

def remove_handle(text): # Remove handles from text
    for link in text:
        result = re.sub(r'@[a-zA-Z_0-9]{1,}', '', text)
        if result:
            return result
        return ''
'''
tweets['text'] = tweets['text'].apply(lambda tweet: remove_handle(tweet))

def remove_hashtags(text): # Remove handles from text
        for link in text:
            result = re.sub(r'#[a-zA-Z_0-9]{1,}', '', text)
            if result:
                return result
            return ''

tweets['text'] = tweets['text'].apply(lambda tweet: remove_hashtags(tweet))

def remove_crap(text): # Remove handles from text
        for link in text:
                result = text.replace('\n','')
                result2 = result.replace('\t','')
                result3 = ' '.join(result2.split())
                if result3:
                    return result3
                return ''
 
tweets['text'] = tweets['text'].apply(lambda tweet: remove_crap(tweet))

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

tweets['text_clean'] = [clean(doc).split() for doc in tweets['text']]
tweets['text_clean'] = tweets['text_clean'].apply(lambda tweet: ' '.join(tweet))
#print(tweets['text_clean'])
'''
senti= vader.SentimentIntensityAnalyzer()
#tweets['sentiment'] = tweets['text_clean'].apply(lambda tweet: senti.polarity_scores(tweet)['compound'])
tweets['sentiment'] = tweets['text'].apply(lambda tweet: senti.polarity_scores(tweet)['compound'])
final = pd.DataFrame()
final['sentiment'] = tweets['sentiment']
final['handles'] = tweets['handles']
final['country'] = tweets['country']
#import gensim
#from gensim import corpora

##Create a data frame - tweet per row
tweets = pd.DataFrame()
tweets["user_id"] = [val1["id"] for val1 in [val2["user"] for val2 in tweets_data]]
tweets["time"] = [val["created_at"] for val in tweets_data]
#tweets["country"] = [val1["country"] for val1 in [val2["place"] for val2 in tweets_data]]
#tweets["coordinates"] = [val1["coordinates"] for val1 in [val2["bounding_box"] for val2 in [val3["place"] for val3 in tweets_data]]]
tweets['coordinates'] = list(map(lambda tweet: tweet['place']['bounding_box']['coordinates']if tweet['place'] != None else None, tweets_data))
#tweets["user_followers"] = [val1["followers_count"] for val1 in [val2["user"] for val2 in tweets_data]]
#tweets["user_friends"] = [val1["friends_count"] for val1 in [val2["user"] for val2 in tweets_data]]

#tweets["text"] = [val["text"] for val in tweets_data]
tweets["language"] = [val["lang"] for val in tweets_data]

tweets.head()

##Find the coordinate centeroids
from shapely.geometry import MultiPoint

tweets["location_centeroid"] = [MultiPoint(val1[0]).centroid if val1 != None else None for val1 in tweets.coordinates]
#tweets["location_centeroid"] = [MultiPoint(val1[0]).centroid for val1 in tweets.coordinates if val1 != None]
#tweets["location_centeroid"] = [MultiPoint(val1[0]).centroid for val1 in tweets.coordinates]
tweets["location_centeroid"] = [val1.coords[0:1] if val1 != None else None for val1 in tweets.location_centeroid]
tweets["location_centeroid"] = [list(val1[0]) if val1 != None else None for val1 in tweets.location_centeroid]

##Find distance from centeroid to each coordinate-set
#from haversine import haversine

l = [[] for i in tweets.location_centeroid]
flatten = lambda l: [item for sublist in l for item in sublist]
'''
for i in range(0, (len(l))):
    for e in range(0,len(flatten(tweets.coordinates[i]))):
        l[i].append(haversine([tweets.ix[i, "location_centeroid"][1], tweets.ix[i, "location_centeroid"][0]],
                              [tweets.iloc[i,3][0][e][1], tweets.iloc[i,3][0][e][0]], miles = True))

tweets["distance_from_centeroid"] = l
for i in range(0, len(tweets["distance_from_centeroid"])):
    tweets.ix[i,"distance_from_centeroid"] = tweets.ix[i,"distance_from_centeroid"][0]
 '''
 
##GEOLOCATION
#from uszipcode import Zipcode
from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode=True)

'''
tweets["zipcode"] = [search.by_coordinates(val1[1], val1[0], radius=r, returns=1) 
                     for val1 in tweets["location_centeroid"] for r in tweets["distance_from_centeroid"]]
'''
final['language'] = tweets['language']
#final['country'] = tweets['country']
tweets["zipcode"] = [search.by_coordinates(val1[1], val1[0], radius=4, returns=1) 
                     if val1 != None else None for val1 in tweets["location_centeroid"]]
lister1 = []
lister2 = []
tweets_zipcode = tweets["zipcode"]

for ijk in range(tweets_zipcode.size):
   entry = tweets_zipcode[ijk]
   if entry != None and len(entry)>0:
       lister1.append(entry[0].county)
   else:
       lister1.append([])
       
for ijk in range(tweets_zipcode.size):
   entry = tweets_zipcode[ijk]
   if entry != None and len(entry)>0:
       lister2.append(entry[0].state)
   else:
       lister2.append([])

final["county"] = lister1
final["state"] = lister2

final.to_csv(final_file, sep='\t',encoding='utf-8')
final_file.close()