import pathlib
import tensorflow as tf
import json
import lc
from .base import Base
import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import utility
from datetime import datetime
import tensorflow_datasets as tfds
 
class Covid19(Base):
    
    def __init__(self, path):
        super().__init__(path, 'covid19')
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.minCount = 3
        self.stopWords = utility.Utility.getStopWords()
        self.validationDataSetSize = 500
        return
    
    def setAllowedPosTypes(self, allowedPOSTypes):
        self.allowedPOSTypes = allowedPOSTypes
        return

    def get(self):
        return self.getTrainingSet(), self.getValidationSet()
        
    def getWholeDataset(self):
        articlesRoot = tf.keras.utils.get_file(self.directoryPath + '/biorxiv_medrxiv/pdf_json/', 
            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz',
            untar=True)

        articlesRoot = pathlib.Path(articlesRoot)
        filePaths = tf.data.Dataset.list_files(str(articlesRoot/'*'))
        
        def read(path):
            label = tf.strings.split(path, '/')[-1]
            data = tf.io.read_file(path)
            return data, label
        
        dataset = filePaths.map(read)
        dataset.shuffle(20000, reshuffle_each_iteration=False).repeat(1)
        return dataset
    
    def getTrainingSet(self):
        dataset = self.getWholeDataset().skip(self.validationDataSetSize)
        if self.totalItems:
            return dataset.take(self.totalItems)
        return dataset
    
    def getValidationSet(self):
        dataset = self.getWholeDataset()
        if self.totalItems:
            return dataset.take(self.totalItems)
        return dataset.take(self.validationDataSetSize)
    
    def getText(self, rawData, includeAbstract = True, updateInfo = False):
        text = None
        data = self.__getData(rawData)
        
        bodyText = [paragraph["text"] for paragraph in data["body_text"]]
        
        if includeAbstract:
            abstractText = [paragraph["text"] for paragraph in data["abstract"]]
            text = ' '.join(abstractText) + ' '.join(bodyText)
        else:
            text = ' '.join(bodyText)
        
        if updateInfo:
            self.processSource(text, False)
        return text
    
    def getTitle(self, rawData):
        data = self.__getData(rawData)
        return data["metadata"]["title"]
    
    def getAbstract(self, rawData):
        data = self.__getData(rawData)
        return data["abstract"]
    
    def getAbstractText(self, rawData, updateInfo = False):
        data = self.__getData(rawData)
        abstractText = [paragraph["text"] for paragraph in data["abstract"]]
        abstractJoinedText = ' '.join(abstractText)
        if updateInfo:
            self.processTarget(abstractJoinedText, False)
        return abstractJoinedText
    
    def getLabel(self, rawData):
        source, label = rawData
        return label.numpy().decode("utf-8")
    
    def getGenerator(self, dataSet, type = 'source'):
        output = None
        if type == 'source':
            output = (self.getText(data, False, True) for data in dataSet)
        else:
            output = (self.getAbstractText(data, True) for data in dataSet)        
        return output
    
    def __getData(self, rawData):
        source, label = rawData
        sourceRaw = source.numpy()
        try:
            sourceRaw = sourceRaw.decode("utf-8", "ignore")
            return json.loads(sourceRaw)
        except Exception as e:
            print(label)
            
        return {}

    def getProcessedText(self, text):
        words = word_tokenize(self.__clean(text))
        allWords = pos_tag(words)
        processedWords = []

        for itemWord in allWords:
            (word, type) = itemWord
            if (type not in self.allowedPOSTypes):
                continue

            word = self._cleanWord(word)
            word = self.stemmer.stem(word.lower())
            processedWords.append(word)

        return ' '.join(processedWords)

    
    def __clean(self, text):
        text = re.sub('<.+?>', '. ', text)
        text = re.sub('&.+?;', '', text)
        text = re.sub('[\']{1}', '', text)
        text = re.sub('[^a-zA-Z0-9\s_\-\?:;\.,!\(\)\"]+', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub('(\.\s*)+', '. ', text)
        return text
