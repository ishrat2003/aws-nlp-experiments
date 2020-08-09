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
from params import Output as OutputParams
from nltk import word_tokenize
from utility import Timer

class Base(Pickle):
    
    def __init__(self, path, name = None, supervised = True):
        self.directoryPath = path
        self.name = name
        self.totalItems = 0
        self.totalValidationItems = 0
        self.splitPercentage = 100
        self.metadata = None
        self.supervised = supervised
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.stopWords = utility.Utility.getStopWords()
        self.stemmer = PorterStemmer()
        self.mode = ''
        self.path = path    
        self.dataCumulativeInfoFile = None
        self.datasetInfo = {}
        self.epochIndex = 1
        return
    
    def setSplitPercentage(self, percentage = 100):
        if ((percentage < 0) or (percentage > 100)):
            percentage = 100
        self.splitPercentage = percentage
        return
    
    def setTotalItems(self, total = 0):
        self.totalItems = total
        return
    
    def setTotalValidationItems(self, validationTotal = 0):
        self.totalValidationItems = validationTotal
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
    
    def startEpoch(self):
        epochName = self.getEpochName()
        Timer.start(epochName)
        return
    
    def saveEpochInfo(self):
        if not self.dataCumulativeInfoFile:
            self.dataCumulativeInfoFile = self.getFile('cumulative', True, self.params['cumulative_directory'])
            
        epochName = self.getEpochName()
        Timer.stop(epochName)
        
        timers = Timer.getFormattedTimers()
        row = {}
        row['dataset'] = self.name
        row['epoch'] = self.epochIndex
        
        for key in timers[epochName].keys():
            row[type + '_' + key] = timers[epochName][key]
        
        for type in self.datasetInfo.keys():
            for key in self.datasetInfo[type].keys():
                row[type + '_' + key] = self.datasetInfo[type][key]
        
        
        self.dataCumulativeInfoFile.write(row)
        self.datasetInfo = {}
        self.epochIndex += 1
        return
    
    def saveInfo(self):
        outputProcessor = OutputParams()
        outputProcessor.addInfo(self.name, self.datasetInfo)
        return
    
    def processSource(self, text, decode = True):
        if decode:
            decodedText = text.decode("utf-8")
        else:
            decodedText = text
            
        if self.mode in ['context', 'masked_context']:
            decodedText = self.__getContext(decodedText)
            text = decodedText.encode("utf-8")
        
        self.__updateInfoBySequenceType(decodedText, 'source')
        return text
    
    def processTarget(self, text, decode = True):
        if decode:
            decodedText = text.decode("utf-8")
        else:
            decodedText = text
        self.__updateInfoBySequenceType(decodedText, 'target')
        return text
    
    def getEpochName(self):
        return 'current_or_last_epoch';
    
    def __updateInfoBySequenceType(self, text, type = 'source'):
        words = word_tokenize(text)
        if ((not words) or (not len(words))):
            return
        
        for key in ['max', 'min', 'avg', 'total']:
            self.__updateInfo(key, len(words), type)
        
        return
    
    
    def __updateInfo(self, key, value, type):
        if (value == 0): 
            # Ignoring edge case
            return
        
        if (type not in self.datasetInfo.keys()):
            self.datasetInfo[type] = {}
            if (type == 'source'):
                self.startEpoch()
                
            
        if (key not in self.datasetInfo[type].keys()):
            if (key == 'total'):
                self.datasetInfo[type][key] = 1
            else:
                self.datasetInfo[type][key] = value
            return
            
        if (key == 'max'):
            if (self.datasetInfo[type][key] < value):
                self.datasetInfo[type][key] = value
                return
                
        if (key == 'min'):
            if (self.datasetInfo[type][key] > value):
                self.datasetInfo[type][key] = value
                return
                
        if (key == 'avg'):
            self.datasetInfo[type][key] = (self.datasetInfo[type][key] + value) / 2
            return
            
        if (key == 'total'):
            self.datasetInfo[type][key] += 1
            return
        return
    
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
        
        if self.mode == 'masked_context':
            return self.maskedContext(contributors)
        
        print('------------------------------------------')
        print('contributors (', str(len(contributors)), '): ', contributors)
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
                tfds.core.ReadInstruction('validation', from_ = 1, to = self.totalValidationItems + 1, unit='abs'),
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