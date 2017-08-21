from pymongo import MongoClient
import re
import ast
from textblob import TextBlob


client = MongoClient()
db = client.zomato
collection = db.reviews

reviewsList = []

rev = db.reviews.find({})

sentences = []

for r in rev:
	x = str(r)
	#x.split('"')
	x = re.findall('".+?"', x)
	x = filter(None, x)
	y = []
	for i in x:
		y = i[1:-2]
		sentences.append(y)
		#y = y.split('(?<!\w\.+\w.)(?<![A-Z][a-z]\.*)(?<=\.*|\?)\s')
		#sentences.append(y)
		#print(y)
		#print('\n')

print(sentences)

noun_phrases = []

for review in sentences:
	blob = TextBlob(review)
	print(blob.noun_phrases)
	noun_phrases.append(blob.noun_phrases)