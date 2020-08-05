import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import utility
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import operator

class Base():


	def __init__(self, text, filterRate = 0, display = False):
		self.rawText = text
		self.text = self.__clean(text)
		self.stopWords = utility.Utility.getStopWords()
		self.stemmer = PorterStemmer()
		self.wordInfo = {}
		self.featuredWordInfo = {}
		self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
		self.minWordSize = 2
		self.sentences = []
		self.punctuationTypes = ['.', '?', '!']
		self.maxCount = 1
		self.maxScore = 0
		self.filterRate = filterRate
		self.topScorePercentage = filterRate
		self.filteredWords = {}
		self.contributors = []
		self.contributingWords = []
		self.display = display
		return


	'''
	allOptions = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	'''
	def setAllowedPosTypes(self, allowedPOSTypes):
		self.allowedPOSTypes = allowedPOSTypes
		return


	def setFilterWords(self, filterRate = 0.2):
		self.filterRate = filterRate
		self.loadFilteredWords()
		return


	def setTopScorePercentage(self, topScorePercentage):
		self.topScorePercentage = topScorePercentage
		return


	def getRawText(self):
		return self.rawText


	def getCleanText(self):
		return self.text


	def getFilteredWords(self):
		return self.filteredWords


	def getContrinutors(self):
		return self.contributors

	def getContributingWords(self):
		self.sortContributers();
		words = [self.contributingWords[index]['pure_word'] for index in self.contributingWords]
		return words

	def getWordInfo(self):
		return self.wordInfo

	def getSentences(self):
		return self.sentences

	def getLengths(self):
		return {
			'sourceLength': len(self.wordInfo),
			'contextLength': len(self.contributingWords)
		}

	def loadFilteredWords(self):
		minAllowedScore = self.maxCount * self.filterRate
		
		self.filteredWords = {}
		for word in self.wordInfo:
			if self.wordInfo[word]['count'] <= minAllowedScore:
				continue

			index = len(self.filteredWords)
			self.filteredWords[word] = self.wordInfo[word]
			self.filteredWords[word]['index'] = index

		if self.display:
			print('----------------------')
			print("Total local vocab: ", len(self.wordInfo))
			print("Filtered local vocab: ", len(self.filteredWords))
			print(self.filteredWords)
		return self.filteredWords


	def loadSentences(self, text):
		words = self.__getWords(text, True)
		self.wordInfo = {}
		self.sentences = []
		currentSentence = []
		for word in words:
			(word, type) = word
			word = self.__cleanWord(word)
			if type in self.punctuationTypes:
				if len(currentSentence) > 1:
					# If more than one word than add as sentence
					self.sentences.append(currentSentence)
				currentSentence = []
			if (len(word) < self.minWordSize) or (word in self.stopWords):
				continue
			
			wordKey = self._addWordInfo(word, type)
			if wordKey and (wordKey not in currentSentence):
				currentSentence.append(wordKey)

        # Processing last sentence
		if len(currentSentence) > 1:
			# If more than one word than add as sentence
			self.sentences.append(currentSentence)

		self.filteredWords = self.wordInfo
		if self.display:
			print(self.filteredWords)
		return self.sentences


	def displayPlot(self, fileName):
		#rcParams['figure.figsize']=15,10
		mpl.rcParams.update({'font.size': 15})
		points = self.getPoints()
		if not points:
			print('No points to display')
			return
	
		plt.figure(figsize=(20, 20))  # in inches(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)[source]
		for point in points:
			plt.scatter(point['x'], point['y'], c = point['color'])
			plt.annotate(point['label'], 
				xy=(point['x'], point['y']), 
				xytext=(5, 2), 
				textcoords='offset points', 
				ha='right', 
				va='bottom')
				
		plt.savefig(fileName)
		print('After saving')
		plt.show()
		return


	def getPoints(self):
		if not len(self.wordInfo):
			return None

		minTopWordScores = self.maxScore * self.topScorePercentage

		points = []
		self.contributors = []
		for word in self.filteredWords:
			point = {}
			point['x'] = self._getX(word)
			point['y'] = self._getY(word)
			point['color'] = 'green'
			point['label'] = self.filteredWords[word]['pure_word']
			point['type'] = self.filteredWords[word]['type']
			
			if self.isTopic(word, minTopWordScores):
				point['color'] = 'red'
				self.contributors.append(word)
				self.contributingWords.append(self.filteredWords[word])

			points.append(point)

		return points


	def isTopic(self, word, minTopWordScores):
		return (self.filteredWords[word]['score'] >= minTopWordScores)

	def sortContributers(self, attribute = 'score'):
		if not len(self.contributingWords):
			return

		sortedContributors = {}

		for value in sorted(self.contributingWords, key=operator.itemgetter(attribute), reverse=True):
			sortedContributors[value['stemmed_word']] = value

		self.contributingWords = sortedContributors
		return

	def _getX(self, word):
		return 0


	def _getY(self, word):
		return 0	


	def _addWordInfo(self, word, type):
		if not word or (type not in self.allowedPOSTypes):
			return None

		if word in self.stopWords:
			return None

		localWordInfo = {}
		localWordInfo['pure_word'] = word
		wordKey = self.stemmer.stem(word.lower())
		localWordInfo['stemmed_word'] = wordKey
		localWordInfo['type'] = type

		if localWordInfo['stemmed_word'] in self.wordInfo.keys():
			self.wordInfo[wordKey]['count'] += 1
			if self.maxCount < self.wordInfo[wordKey]['count']:
				self.maxCount = self.wordInfo[wordKey]['count']
			return wordKey

		localWordInfo['count'] = 1
		localWordInfo['index'] = len(self.wordInfo)
		self.wordInfo[wordKey] = localWordInfo

		return wordKey


	def __getWords(self, text, tagPartsOfSpeach = False):
		words = word_tokenize(text)

		if tagPartsOfSpeach:
			return pos_tag(words)

		return words


	def __cleanWord(self, word):
		return re.sub('[^a-zA-Z0-9]+', '', word)


	def __clean(self, text):
		text = re.sub('<.+?>', '. ', text)
		text = re.sub('&.+?;', '', text)
		text = re.sub('[\']{1}', '', text)
		text = re.sub('[^a-zA-Z0-9\s_\-\?:;\.,!\(\)\"]+', ' ', text)
		text = re.sub('\s+', ' ', text)
		text = re.sub('(\.\s*)+', '. ', text)
		return text

