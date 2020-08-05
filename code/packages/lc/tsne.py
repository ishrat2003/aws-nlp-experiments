import numpy
from . import Base
from sklearn.manifold import TSNE
import io, os

class TSNELC(Base):

	def __init__(self, text, filterRate = 0.2):
		super().__init__(text, filterRate)
		self.loadSentences(text)
		self.perplexity = 3
		self.numberOfComponents = 2
		self.numberOfIterations = 250
		self.learnedEmbeddings = None
		self.markedWords = []
		return

	def setMarkedWords(self, words):
		self.markedWords = words
		return


	def setPerplexity(self, perplexity):
		self.perplexity = perplexity
		return


	def setNumberOfComponents(self, numberOfComponents):
		self.numberOfComponents = numberOfComponents
		return

	
	def setNumberOfIterations(self, numberOfIterations):
		self.numberOfIterations = numberOfIterations
		return


	def getWordCoOccurenceVectors(self, path = None):
		vocabSize = len(self.wordInfo)
		vectors = numpy.zeros((vocabSize, vocabSize))
		
		filteredWords = self.filteredWords.keys()

		for sentence in self.sentences:
			for word1 in sentence:
				word1 = self.stemmer.stem(word1.lower())
				if word1 not in filteredWords:
					continue
				for word2 in sentence:
					if (word1 == word2) or (word2 not in filteredWords):
						continue
					word1Index = self.wordInfo[word1]['index']
					word2Index = self.wordInfo[word2]['index']
					vectors[word1Index][word2Index] += 1

		if path:
			self.saveVectorAndMeta(path, vectors)
			
		return vectors

	def saveVectorAndMeta(self, path, vectors):
		outV = io.open(os.path.join(path, 'vecs.tsv'), 'w', encoding='utf-8')
		outM = io.open(os.path.join(path, 'meta.tsv'), 'w', encoding='utf-8')
		outM.write("word\tcategory\n")
		
		for word in self.filteredWords.keys():
			vec = vectors[self.filteredWords[word]['index']]
			outV.write('\t'.join([str(x) for x in vec]) + "\n")
			category = "green"
			if self.isTopic(word, None):
				category = "red"
			outM.write(self.filteredWords[word]['pure_word'] + "\t"  + category + "\n")	
		outV.close()
		outM.close()
		print('Word vector saved')
		return
     
	def train(self, path = None):
		vectors = self.getWordCoOccurenceVectors(path)
		tsne = TSNE(perplexity = self.perplexity, 
			n_components = self.numberOfComponents, 
			init = 'pca', 
			n_iter = self.numberOfIterations, 
			method='exact')
		self.learnedEmbeddings = tsne.fit_transform(vectors)
		return


	def isTopic(self, word, topWordScores):
		if word in self.markedWords:
			return True
		return False


	def _getX(self, word):
		index = self.filteredWords[word]['index']
		return self.learnedEmbeddings[index, 0]


	def _getY(self, word):
		index = self.filteredWords[word]['index']
		return self.learnedEmbeddings[index, 1]