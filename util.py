import numpy as np 
import string
import nltk

def init_weights(M1, M2):
	W = np.random.randn(M1,M2)/np.sqrt(M1+M2)
	return W.astype(np.float32)

def classification_rate(T, P):
	return np.mean(T==P)

def remove_punctuation(s):
	translator = str.maketrans('', '', string.punctuation)
	s = s.translate(translator)
	return s

def get_robert_frost_data():
	df = open("robert_frost.txt")

	text = [sentence.strip() for sentence in df]
	#print(len(text[0]))

	word2idx = {'START':0, 'END':1}
	curIdx = 2
	sentences = []
	for sentence in text:
		tokens = remove_punctuation(sentence.lower()).split()
		if tokens:
			vec = []
			for word in tokens:
				if word not in word2idx:
					word2idx[word] = curIdx
					curIdx +=1
				vec += [word2idx[word]]
			sentences.append(vec)

	return sentences, word2idx


def get_classifier_data():
	rf_data = open('robert_frost.txt')
	sentences = [sentence.strip() for sentence in rf_data]
	sentences = [remove_punctuation(sentence.lower()) for sentence in sentences]
	print(sentences[0])	
	print(nltk.pos_tag(sentences[0].split()))

#get_classifier_data()
