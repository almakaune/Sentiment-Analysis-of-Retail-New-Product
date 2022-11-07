#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:08:15 2020

@author: almaaune
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from wordcloud import WordCloud, STOPWORDS

import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer


# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt



df = pd.read_csv('Aritzia_New_SS_2020_Reviews_Topic_model.csv')

print(df.columns)



# PREPROCESSING
# --------------

# convert None to NaN
df['Rating'] = df['Rating'].replace('None', np.nan)
df['Review'][182] = 'None'

print(df.tail())

# convert rating to numerical
df['Rating'] = pd.to_numeric(df['Rating'])

# convert text to lower case.
df['Review'] = [x.lower() for x in df['Review']]

print(df.info())

# drop entries where rating = nan
df.dropna(subset = ['Rating'], inplace = True)

# EDA
# ----

# what is the total review count by clothing type?
review_count_clothing = df['Clothing'].value_counts()
review_count_clothing = review_count_clothing.reset_index()

print(review_count_clothing)

plt.figure(figsize = (7,7))
plt.bar(review_count_clothing['index'], review_count_clothing['Clothing'])
plt.xticks(rotation = 45)

# what is the average rating by clothing type?
avg_rating_clothing = df.groupby('Clothing')['Rating'].mean().round(2)
print(avg_rating_clothing)

# what is the total review count by brand?
review_count_brand = df['Brand'].value_counts()
review_count_brand = review_count_brand.reset_index()

print(review_count_brand)

plt.figure(figsize = (7,7))
plt.bar(review_count_brand['index'], review_count_brand['Brand'])
plt.xticks(rotation=45)


# what is average rating by brand?
avg_rating_brand = df.groupby('Brand')['Rating'].mean().round(2)

print(avg_rating_brand)


# what is the rating distribution
ratings = df['Rating'].value_counts()
ratings = ratings.reset_index()

plt.figure(figsize = (7,7))
plt.bar(ratings['index'], ratings['Rating'])

print(ratings)




# WORD CLOUD
# -----------

# OVERALL REVIEWS
text = " ".join(review for review in df.Review)

# POSITIVE REVIEWS (4-5)
pos_rev = df[df['Rating'] >=4]
pos = " ".join(review for review in pos_rev.Review)

# NEGATIVE REVIEWS (1-3)
neg_rev = df[df['Rating'] < 4]
neg = " ".join(review for review in neg_rev.Review)


# create stopword list:
stopwords = set(STOPWORDS)

stopwords.update(['read','got','way','see','completely','great','super', 'one','expected','look','wear',
                  'love','perfect','cute', 'really','nice','much'])



# Overall wordcloud
print("OVERALL WORDCLOUD")
wordcloud = WordCloud(stopwords = stopwords, background_color = 'white').generate(text)
plt.figure()

# display the generated image
plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')
plt.show()


# Positive wordcloud
print("POSITIVE WORDCLOUD")
wordcloud_pos = WordCloud(stopwords = stopwords, background_color = 'white').generate(pos)
plt.figure()

# display the generated image
plt.imshow(wordcloud_pos, interpolation = 'bilinear')

plt.axis('off')
plt.show()



# Negative workdcloud
print("NEGATIVE WORDCLOUD")
wordcloud_neg = WordCloud(stopwords = stopwords, background_color = 'white').generate(neg)
plt.figure()

# display the generated image
plt.imshow(wordcloud_neg, interpolation = 'bilinear')

plt.axis('off')
plt.show()



# LDA MODEL

# create list of punctuation marks
import string
import re
punctuations = string.punctuation
print(punctuations)

# create list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# load englisth tokeniger, tagger, parser NER and word vectors
parser = English()

# create tokenizer function
def spacy_tokenizer(sent):
    # create token object which is used to create documents with linguistic annotations
    mytokens = parser(sent)
    # lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in mytokens]
    # removing stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    # return preprocessed list of tokens
    return mytokens
 

# basic function to remove spaces and convert text into lowercase
def clean_text(text):
       
        return text.strip().lower()

text = spacy_tokenizer(text)

text = [clean_text(w) for w in text]

# remove more stop words: 
stopwords = ['…', 'read', 'way','maybe','great','bit','super','fits','cut','true','usually','little'
             'small','beautiful','nicely','like','cute','love','good','aritzia','light','loved','got','slightly',
             'colors','size','wear','easy','look','buy','need','going','4','0','definitely','amazing','2','nice',
             'typically','version','bought','perfectly','perfect','run','ordered','body','best','obsessed','tried','absolutely'
             ,'new','tried','want','extra','purchased','think','pretty','feels','expected','wearing','s']


new_text = [w for w in text if w not in stopwords]

new_text = np.array(new_text)

# create dictionary
dictionary = corpora.Dictionary([new_text])

# term document frequency
corpus = [dictionary.doc2bow(t.split()) for t in new_text]

print(corpus[:1])
print(dictionary[312])

# build LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                           id2word = dictionary,
                                           num_topics= 10,
                                           random_state = 100,
                                           passes = 10)

topics = ldamodel.print_topics(num_words = 5, num_topics = 10)

for line in topics:
    print(line)
    print()


# visualize topics 
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus,dictionary)
vis

"""
# instantiate CountVectorizer
vect0 = CountVectorizer(min_df = 20, max_df = 0.2, stop_words = 'english', token_pattern='(?u)\\b\\w\\w\\w+\\b')

# fit transform using newsgroup_data
X0 = vect0.fit_transform(df['Review'])

# convert sparse matrix to gensim corpus
corpus0 = gensim.matutils.Sparse2Corpus(X0, documents_columns = False)

# mapping from word IDs to words
id_map0 = dict((v,k) for k,v in vect0.vocabulary_.items())

# create ldamodel
ldamodel0 = LdaModel(corpus0, id2word=id_map0, passes = 25, random_state = 34, num_topics = 10)

# get topics
topics = ldamodel0.print_topics(num_words = 5, num_topics = 10)

for i in range(len(topics)):
    print(topics[i])
    print
    
    
(0, '0.480*"really" + 0.287*"nice" + 0.097*"like" + 0.075*"fabric" + 0.026*"wear"')
(1, '0.756*"cardigan" + 0.117*"cute" + 0.045*"flattering" + 0.042*"fabric" + 0.012*"really"')
(2, '0.541*"hoodie" + 0.233*"light" + 0.084*"fabric" + 0.081*"nice" + 0.027*"got"')
(3, '0.358*"flattering" + 0.299*"pants" + 0.230*"wear" + 0.057*"cute" + 0.024*"comfortable"')
(4, '0.785*"small" + 0.143*"wear" + 0.017*"great" + 0.005*"hoodie" + 0.004*"got"')
(5, '0.403*"cute" + 0.354*"super" + 0.089*"comfy" + 0.076*"wear" + 0.058*"fabric"')
(6, '0.637*"fits" + 0.270*"got" + 0.047*"great" + 0.015*"comfy" + 0.004*"wear"')
(7, '0.408*"comfortable" + 0.348*"material" + 0.106*"nice" + 0.090*"wear" + 0.018*"really"')
(8, '0.684*"great" + 0.149*"pants" + 0.129*"fabric" + 0.015*"comfortable" + 0.003*"light"')
(9, '0.425*"like" + 0.314*"comfy" + 0.119*"got" + 0.110*"material" + 0.002*"hoodie"')
"""
