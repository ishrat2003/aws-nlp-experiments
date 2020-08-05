import os
import tensorflow_datasets as tfds
import utility
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import lc as LC
from file.pickle import Pickle

class Base(Pickle):
    
    def __init__(self, path, name = None, supervised = True):
        self.directoryPath = path
        self.name = name
        self.totalItems = 0
        self.splitPercentage = 100
        self.metadata = None
        self.supervised = supervised
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.stopWords = utility.Utility.getStopWords()
        self.stemmer = PorterStemmer()
        self.mode = ''
        self.path = path    
        self.sourceLengthFile = self.getFile('sourceLength.csv')
        self.targetLengthFile = self.getFile('targetLength.csv')
        return
    
    def setSplitPercentage(self, percentage = 100):
        if ((percentage < 0) or (percentage > 100)):
            percentage = 100
        self.splitPercentage = percentage
        return
    
    def setTotalItems(self, total = 100):
        self.totalItems = total
        return
    
    def setMode(self, mode = ''):
        self.mode = mode
        return
    
    def setParams(self, params):
        self.params = params
        return
    
    def get(self):
        self.dlConfig = tfds.download.DownloadConfig(manual_dir=self.directoryPath)
        return self.dataset()
    
    def getPath(self):
        return self.path
    
    def getTrainingSet(self):
        trainingSet, validationSet = self.get()
        return trainingSet
    
    def getMetadata(self):
        return self.metadata
    
    def getProcessedPath(self):
        return self.path
    
    def getVocabPath(self, prefix = ''):
        print(os.path.join(self.path, prefix + "vocab"))
        return os.path.join(self.path, prefix + "vocab")

    def dataset(self):
        return None, None
    
    def getGenerator(self, trainingSet, type = 'source'):
        return None
    
    def getText(self, rawData):
        text, summary = rawData
        return text.numpy().decode('utf-8')
    
    def getAbstract(self, rawData):
        text, summary = rawData
        return summary.numpy().decode('utf-8')
    
    def getTitle(self, rawData):
        return self.getAbstract(rawData)
    
    def getLabel(self, rawData):
        return 'Todo'
    
    def getAbstract(self, rawData):
        text, summary = rawData
        return summary.numpy()
    
    def getPath(self):
        return self.path
    
    def getProcessedText(self, text):
        words = word_tokenize(text)
        allWords = pos_tag(words)
        processedWords = []

        for itemWord in allWords:
            (word, type) = itemWord
            if (type not in self.allowedPOSTypes) or (word in self.stopWords):
                continue

            word = self._cleanWord(word)
            word = self.stemmer.stem(word.lower())
            processedWords.append(word)

        return ' '.join(processedWords)
    
    def processSource(self, text):
        # print('------------------------------------------')
        # print('Text:  ', text);
        if self.mode in ['context', 'masked_context']:
            text = text.decode("utf-8")   
            context = self.__getContext(text)
            return context.encode("utf-8")
        return text
    
    def processTarget(self, text):
        # print('Summary:    ', text)
        words = word_tokenize(text.decode("utf-8"))
        self.targetLengthFile.write({
            'targetLength': len(words)
        })
        return text
    
    def __getContext(self, text):
        peripheralProcessor = LC.Peripheral(text)
        peripheralProcessor.setAllowedPosTypes(self.allowedPOSTypes)
        peripheralProcessor.setPositionContributingFactor(5)
        peripheralProcessor.setOccuranceContributingFactor(0)
        peripheralProcessor.setProperNounContributingFactor(0)
        peripheralProcessor.setTopScorePercentage(0.2)
        peripheralProcessor.setFilterWords(0)
        peripheralProcessor.loadSentences(text)
        peripheralProcessor.loadFilteredWords()
        peripheralProcessor.train()
        points = peripheralProcessor.getPoints()
        
        contributors = peripheralProcessor.getContributingWords()
        self.sourceLengthFile.write(peripheralProcessor.getLengths())
        
        if self.mode == 'masked_context':
            return self.maskedContext(contributors)
        
        # print('------------------------------------------')
        # print('contributors (', str(len(contributors)), '): ', contributors)
        return ' '.join(contributors)
    
    def maskedContext(self, contributors):
        maskedContributors = ['CONTEXT' + str(index) for index in contributors]
        # print('------------------------------------------')
        # print('masked contributors (', str(len(contributors)), '): ', contributors)
        return ' '.join(maskedContributors)
    
    def _cleanWord(self, word):
        return re.sub('[^a-zA-Z0-9]+', '', word)
    
    def _load(self, key): 
        if self.totalItems:
            readInstructions = [
                tfds.core.ReadInstruction('train', from_ = 1, to = self.totalItems + 1, unit='abs'),
                tfds.core.ReadInstruction('validation', from_ = 1, to = self.totalItems + 1, unit='abs'),
            ]
        elif self.splitPercentage:
            readInstructions = [
                tfds.core.ReadInstruction('train', to = self.splitPercentage, unit='%'),
                tfds.core.ReadInstruction('validation', to = self.splitPercentage, unit='%'),
            ]
        else:
            readInstructions = ['train', 'validation']  
            
        data, self.metadata = tfds.load(key, 
            data_dir = self.getProcessedPath(), 
            with_info = True, 
            as_supervised = self.supervised, 
            download_and_prepare_kwargs = dict(download_config=self.dlConfig),
            split = readInstructions)
        
        return data[0], data[1]
    