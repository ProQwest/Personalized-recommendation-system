import os
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

from gensim.models import word2vec


files_pos = os.listdir('/Users/prady/Desktop/major-project/restaurant/positive')
files_neg = os.listdir('/Users/prady/Desktop/major-project/restaurant/negative')

raw_sentences = []
sentences = []


for filename in files_pos:
    path = os.path.join('/Users','prady','Desktop','major-project','restaurant','positive',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())

for filename in files_neg:
    path = os.path.join('/Users','prady','Desktop','major-project','restaurant','negative',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())
'''
files_pos = os.listdir('/Users/prady/Desktop/major-project/review_polarity/txt_sentoken/pos')
files_neg = os.listdir('/Users/prady/Desktop/major-project/review_polarity/txt_sentoken/neg')

for filename in files_pos:
    path = os.path.join('/Users','prady','Desktop','major-project','review_polarity','txt_sentoken','pos',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())

for filename in files_neg:
    path = os.path.join('/Users','prady','Desktop','major-project','review_polarity','txt_sentoken','neg',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())
'''
#print(sentences)

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", str(review))
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

for s in raw_sentences:
	sentences.append(review_to_wordlist(s))


#print(sentences[0])

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print("Training model...")

model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample=downsampling)

model.init_sims(replace=True)

model_name = "review_polarity_model_restaurant"
model.save(model_name)

