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
        if i % 10000 == 0:      
            print ('\nWe have gathered:',len(tweets_data), 'tweets.\n')
            print('This is one of them:', tweet['text'])
        if i >= 1000: # Define size of subset
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
tweets['time'] =    list(map(lambda tweet: tweet['created_at'], tweets_data))
#tweets['lang'] =    list(map(lambda tweet: tweet['lang'], tweets_data))
#tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))

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
        #print ("\n\nLink free:\n",result)
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
        try:
            result = re.sub(r'@[a-zA-Z_0-9]{4,}', '', text)
            #print ("\n\nHandle free:\n", result)
            return result
        except:
            pass


#%%          

#re.match returns None if it cannot find a match. Probably the cleanest solution to this problem is to just do this:

# There is no need for the try/except anymore
match = re.match(r'^(\S+) (.*?) (\S+)$', full)
if match is not None:
    clean = filter(None, match.groups())
else:
    clean = ""
#Note that you could also do if match:, but I personally like to do 
# if match is not None: because it is clearer. 
# "Explicit is better than implicit" remember. ;)

#%%
    
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
    if doc is not None:
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        #normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        normalized = " ".join(word for word in punc_free.split())
        return normalized
    else:
        doc = ""
        return doc

#%%
# This is the clean corpus.
    # This part is not working anymore: 'NoneType' object has no attribute 'lower'
    # I think the problem occurs when I remove twitter handles and hyperlinks
    # because this means that some cells are NA's, which returns an empty object

tweets['text_clean'] = [clean(doc).split() for doc in tweets['text']] # Check df: ISIS is lemmatized to isi which might be problem

# change "isi" to isis (this doesnt matter for the sentiment analysis but it will for the topic modelling)

#%%

#from textblob import TextBlob # Textblob for sentiment analysis

tweets['text_clean'] = tweets['text_clean'].apply(lambda tweet: ' '.join(tweet))
tweets['blob_sentiment'] = tweets['text_clean'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Run two different sentiment analysis and correlate the outcomes..
# Find model that includes emojis

#%%
from nltk.sentiment import vader
import seaborn as sns

senti= vader.SentimentIntensityAnalyzer()

#def vader_polarity(text):
    #""" Transform the output to a binary 0/1 result """
    #score = senti.polarity_scores(text)
    #return 1 if score['pos'] > score['neg'] else 0

#tweets['vader_sentiment'] = tweets['text_clean'].apply(lambda tweet: vader_polarity(tweet)['compound'])

#%%

tweets['text_clean'] = tweets['text_clean'].apply(lambda tweet: ' '.join(tweet))
tweets['vader_sentiment'] = tweets['text_clean'].apply(lambda tweet: senti.polarity_scores(tweet)['compound'])

# Impliment SentiStrength (better for short texts) 
# Website: http://sentistrength.wlv.ac.uk/

#%%

#tweets[['blob_sentiment', 'vader_sentiment']].corr(method='pearson') # 0.52
tweets[['blob_sentiment', 'vader_sentiment']].corr(method='spearman') # 0.50


#sns.heatmap(corrmatrix) # Correlations below 0.6... 

#%%

# Append state onto dataframe

df = pd.read_table('/Users/Daniel/Desktop/final_frame_full.txt', header=0)

#%%

tweets['state'] = df['state']
tweets['county'] = df['county']
tweets['vader_sentiment2'] = df['sentiment']
tweets['language'] = df['language']
tweets['country'] = df['country']

#%%
#del df['Unnamed: 0'], df['handles'], df['sentiment'], df['country'], df['language'], df['county']

#tweets['state'] = df['state']

#del df

#%%

# Make subset with selected states for topic modelling

NY = tweets[tweets.state == 'NY'] # New York: Its a solely democratic state that went for Hillary Clinton with a big margin
TN = tweets[tweets.state == 'TN'] 
# Same as New York, but the other way around... 


#%%
del NY['vader_sentiment'], NY['time'], NY['state'], NY['country'], NY['vader_sentiment2'], NY['language'], NY['county'] 
del TN['vader_sentiment'], TN['time'], TN['state'], TN['country'], TN['vader_sentiment2'], TN['language'], TN['county'] 

NY = pd.DataFrame(NY)
TN = pd.DataFrame(TN)

#%%

def clean(doc):
    if doc is not None:
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        normalized = " ".join(word for word in punc_free.split())
        return normalized
    else:
        doc = ""
        return doc

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))

# Create a set of punctuation words 
exclude = set(string.punctuation) 

# This is the function makeing the lemmatization
lemma = WordNetLemmatizer()

NY['text_clean'] = [clean(doc).split() for doc in NY['text']]
TN['text_clean'] = [clean(doc).split() for doc in TN['text']]

#%%


# If you cut this session the datasets are saved in Fundamentals of data-science

# Find model fit for corpus

# Find way of running topic-model with bigrams?
# Hot vectors?

from gensim import corpora, models

texts_for_lda = NY['text_clean'] # Taking stopped part of corpus for LDA modelling

texts = texts_for_lda
id2word = corpora.Dictionary(texts)
id2word.filter_extremes(no_below=5, no_above=0.5) # Remove words ocurring in less than 5 and more than 50% of docs

mm = [id2word.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(mm) # Term frequency-inverse document frequency
lda_NY = models.ldamodel.LdaModel(corpus = tfidf[mm], id2word = id2word, num_topics = 2, alpha = "auto")

lda_NY.print_topics(num_words=10)[:5] # Check if it worked

texts_for_lda = TN['text_clean'] # Taking stopped part of corpus for LDA modelling

texts = texts_for_lda
id2word = corpora.Dictionary(texts)
id2word.filter_extremes(no_below=5, no_above=0.5) # Remove words ocurring in less than 5 and more than 50% of docs

mm = [id2word.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(mm) # Term frequency-inverse document frequency
lda_TN = models.ldamodel.LdaModel(corpus = tfidf[mm], id2word = id2word, num_topics = 5, alpha = "auto")

lda_TN.print_topics(num_words=10)[:5] # Check if it worked


#%%

# get term relevance

corpora_dict = corpora.Dictionary(NY['text_clean'])
corpus = [corpora_dict.doc2bow(t) for t in NY['text_clean']]

viz = pyLDAvis.prepare(lda_NY, corpus, dictionary, sort_topics=False)

name_dict = {   0: "1", # 1 on the chart
                1: "1",    # 2 on the chart
                2: "2",  # 3 on the chart
                3: "3",
                4: "4",
                5: "5"
            }

for_viz = {}

# specify parameter
lambda_ = 0.4

viz_data = viz.topic_info
viz_data['relevance'] = lambda_ * viz_data['logprob'] + (1 - lambda_) * viz_data['loglift']

# plot the terms
plt.rcParams['figure.figsize'] = [20, 11]
fig, ax_ = plt.subplots(nrows=1, ncols=3)
ax = ax_.flatten()

for j in range(lda_NY.num_topics):
    df = viz.topic_info[viz.topic_info.Category=='Topic'+str(j+1)].sort_values(by='relevance', ascending=False).head(30)

    df.set_index(df['Term'], inplace=True)
    sns.barplot(y="Term", x="Freq",  data=df, ax = ax[j])
    sns.set_style({"axes.grid": False})

    ax[j].set_xlim([df['Freq'].min()-1, df['Freq'].max()+1])
    ax[j].set_ylabel('')
    ax[j].set_title(name_dict[j], size=15)
    ax[j].tick_params(axis='y', labelsize=13)

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

#%%
# This is the clean corpus.
    # This part is not working anymore: 'NoneType' object has no attribute 'lower'
    # I think the problem occurs when I remove twitter handles and hyperlinks
    # because this means that some cells are NA's, which returns an empty object

tweets['text_clean'] = [clean(doc).split() for doc in tweets['text']

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
