import os
import tensorflow_datasets as tfds
from .base import Base
from .newsroom import Newsroom
from .ptToEnTranslate import PtToEnTranslate
from .scientificPapers import ScientificPapers
from .cnnDailyMail import CNNDailyMail
from .wikiHow import WikiHow
from .multiNews import MultiNews
from .covid19 import Covid19
from .pharmaNews import PharmaNews
from .bhot import BHOT
from .brexit import Brexit

class Core:

    def __init__(self, name, path, params = {}):
        self.name = name
        self.path = path
        self.dataSetProcessor = Base(self.path, self.name)
        self.params = params
        return
    
    def get(self, percentage = 100, total = 0):
        if (self.name == 'newsroom'):
            self.dataSetProcessor = Newsroom(self.path)
        elif (self.name == 'pt_to_en_translate'):
            self.dataSetProcessor = PtToEnTranslate(self.path)
        elif (self.name == 'scientific_papers_arxiv'):
            self.dataSetProcessor = ScientificPapers(self.path, 'scientific_papers/arxiv')
        elif (self.name == 'scientific_papers_pubmed'):
            self.dataSetProcessor = ScientificPapers(self.path, 'scientific_papers/pubmed')
        elif (self.name == 'cnn_dailymail'):
            self.dataSetProcessor = CNNDailyMail(self.path)
        elif (self.name == 'wikihow'):
            self.dataSetProcessor = WikiHow(self.path)
        elif (self.name == 'multi_news'):
            self.dataSetProcessor = MultiNews(self.path)
        elif (self.name == 'covid19'):
            self.dataSetProcessor = Covid19(self.path)
        elif (self.name == 'pharma_news'):
            self.dataSetProcessor = PharmaNews(self.path)
        elif (self.name == 'bhot'):
            self.dataSetProcessor = BHOT(self.path)
        elif (self.name == 'brexit'):
            self.dataSetProcessor = Brexit(self.path)
        else:
            return None

        self.dataSetProcessor.setSplitPercentage(percentage)
        self.dataSetProcessor.setTotalItems(total)
        self.dataSetProcessor.setMode(self.params['mode']);
        self.dataSetProcessor.setParams(self.params);
        return self.dataSetProcessor
