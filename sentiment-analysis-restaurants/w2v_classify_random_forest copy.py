from gensim.models import Word2Vec
import numpy as np


model = Word2Vec.load("/Users/prady/Desktop/major-project/review_polarity_model")

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
 
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", str(review))
    #
    # Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    #Return a list of words
    return(words)

for s in raw_sentences:
	sentences.append(review_to_wordlist(s))

def makeFeatureVec(words, model, num_features):

    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    #if(len(words)==0):
    #	print("null review")
    #
    for word in words:
    	#print(word)
    	if str(word) in index2word_set: 
	    	nwords = nwords + 1.
	        featureVec = np.add(featureVec,model[str(word)])
    # Divide the result by the number of words to get the average
    if(nwords==0):
    	print(featureVec)
    	return featureVec
    featureVec = np.divide(featureVec,nwords)
    #print(featureVec)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

num_features = 300

from itertools import chain
train_reviews = list(chain(sentences[0:700], sentences[1000:1700]))
test_reviews = list(chain(sentences[700:1000], sentences[1700:2000]))

trainDataVecs = getAvgFeatureVecs( train_reviews, model, num_features )
testDataVecs = getAvgFeatureVecs( test_reviews, model, num_features )



train_output = []
i=0
while(i<1400):
	if (i<700):
		train_output.append(1)
	else:
		train_output.append(0)
	i=i+1

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

forest = forest.fit( trainDataVecs, train_output )

# Test & extract results 
result = forest.predict( testDataVecs )

result = result.tolist()
#print(result)
#print(len(result))
#print(type(result))
'''
cov_matrix = np.cov(trainDataVecs.T)

for i in range(300):
	for j in range(300):
		if(i==j):
			cov_matrix[i][j]= 9999
		else:
			cov_matrix=abs(cov_matrix)
#print(np.argmax(np.cov(trainDataVecs.T)))
print(np.unravel_index(cov_matrix.argmin(), cov_matrix.shape))
'''

import matplotlib
from matplotlib import pyplot as plt

fig = plt.figure()

plt.scatter(trainDataVecs[0:700,109],trainDataVecs[0:700,165], c='green',marker='+')
plt.scatter(trainDataVecs[700:1400,109],trainDataVecs[700:1400,165], c= 'red', marker='o')
plt.show()


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
