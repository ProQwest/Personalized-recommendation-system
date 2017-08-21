from sklearn.cluster import KMeans
import time
from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("/Users/prady/Desktop/major-project/review_polarity_model")

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.index2word, idx ))

# For the first 10 clusters
for cluster in xrange(0,10):
    #
    # Print the cluster number  
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print words

def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

import os
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

from gensim.models import word2vec


files_pos = os.listdir('/Users/prady/Desktop/review_polarity/txt_sentoken/pos')
files_neg = os.listdir('/Users/prady/Desktop/review_polarity/txt_sentoken/neg')

raw_sentences = []
sentences = []

for filename in files_pos:
    path = os.path.join('/Users','prady','Desktop','major-project','review_polarity','txt_sentoken','pos',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())

for filename in files_neg:
    path = os.path.join('/Users','prady','Desktop','major-project','review_polarity','txt_sentoken','neg',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())

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

from itertools import chain
train_reviews = list(chain(sentences[0:700], sentences[1000:1700]))
test_reviews = list(chain(sentences[700:1000], sentences[1700:2000]))

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (1400, num_clusters), dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( 600, num_clusters), dtype="float32" )

counter = 0
for review in test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1


train_output = []
i=0
while(i<1400):
	if (i<700):
		train_output.append(1)
	else:
		train_output.append(0)
	i=i+1

from sklearn.svm import SVC
clf = SVC(kernel='linear',C=0.01)
clf.fit(train_centroids, np.asarray(train_output))
result = clf.predict(test_centroids)
result = result.tolist()

true_positives = 0.0
true_negatives = 0.0
false_positives = 0.0
false_negatives = 0.0

for i in range(600):
	if(i<300):
		if(result[i]==1):
			true_positives=true_positives+1
		else:
			false_negatives=false_negatives+1
	else:
		if(result[i]==0):
			true_negatives=true_negatives+1
		else:
			false_positives=false_positives+1


precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives+false_negatives)
f_measure = 2*precision*recall/(precision+recall)

print("true_positives")
print(true_positives)
print("true_negatives")
print(true_negatives)
print("false_positives")
print(false_positives)
print("false_negatives")
print(false_negatives)
print("precision = ")
print(precision)
print("recall = ")
print(recall)
print("f_measure = ")
print(f_measure)
