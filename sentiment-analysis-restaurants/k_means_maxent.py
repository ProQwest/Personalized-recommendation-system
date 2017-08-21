from sklearn.cluster import KMeans
import time
from gensim.models import Word2Vec
import numpy as np
from nltk.classify.maxent import MaxentClassifier


#listOfFiles=[]

model = Word2Vec.load("/Users/prady/Major Project/sentiment-analysis-restaurants/review_polarity_model_restaurant")

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


files_pos = os.listdir('/Users/prady/Major Project/restaurant/positive')
files_neg = os.listdir('/Users/prady/Major Project/restaurant/negative')

raw_sentences = []
sentences = []

for filename in files_pos:
    path = os.path.join('/Users','prady','Major Project','restaurant','positive',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())
    #print(filename)
    #listOfFiles.append(filename)

for filename in files_neg:
    path = os.path.join('/Users','prady','Major Project','restaurant','negative',filename)
    with open(path) as current_file:
        raw_sentences.append(current_file.readlines())
#listOfFiles.append(filename)
#print(sentences)

print(len(raw_sentences))

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
train_reviews = list(chain(sentences[0:795], sentences[1135:1930]))
test_reviews = list(chain(sentences[795:1135], sentences[1930:3045]))

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (1590, num_clusters), dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( 1455, num_clusters), dtype="float32" )

counter = 0
for review in test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1


train_output = []
i=0
while(i<1590):
	if (i<795):
		train_output.append(1)
	else:
		train_output.append(0)
	i=i+1
'''
from sklearn.ensemble import RandomForestClassifier
# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids, train_output)
result = forest.predict(test_centroids)
'''

train_set=[]
for i in range(1590):
    f = train_centroids[i]
    f = f.tolist()
    output = train_output[i]
    train_set.append((dict(enumerate(f)), output))

maxent_classifier = MaxentClassifier.train(train_set, max_iter=5)
true_positives = 0.0
true_negatives = 0.0
false_positives = 0.0
false_negatives = 0.0

#print listOfFiles[0]

#print listOfFiles[1]

for i in range(1455):
    input_feature = test_centroids[i].tolist()
    prediction = maxent_classifier.classify(dict(enumerate(input_feature)))
    if(i<340):
	if(prediction==1):
		print("****************************************************")
		print("True positive")
		temp = i+795
		temp = str(temp) + '.txt'
		mypath = os.path.join('/Users','prady','Major Project','restaurant','positive',temp)
		with open(mypath,'r') as fin:
			print fin.read()
		print("****************************************************")
		true_positives=true_positives+1
	else:
	  false_negatives=false_negatives+1
	  print("****************************************************")
	  print("False negative")
	  temp = i+795
	  temp = str(temp) + '.txt'
	  mypath = os.path.join('/Users','prady','Major Project','restaurant','positive', temp)
	  with open(mypath,'r') as fin:
		  print fin.read()
	  print("****************************************************")
		
    else:
	if(prediction==0):
		print("****************************************************")
		print("True negative")
		temp = i-340+795
		temp = str(temp) + '.txt'
		mypath = os.path.join('/Users','prady','Major Project','restaurant','negative',temp)
		with open(mypath) as fin:
			print fin.read()
		print("****************************************************")
		true_negatives=true_negatives+1
	else:
	  print("****************************************************")
	  print("False positive")
	  temp = 795+i-340
	  false_positives=false_positives+1
	  temp = str(temp) + '.txt'
	  mypath = os.path.join('/Users','prady','Major Project','restaurant','negative',temp)
	  with open(mypath) as fin:
		  print fin.read()
	  print("****************************************************")

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
