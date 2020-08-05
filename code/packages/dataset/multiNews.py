from .base import Base
import tensorflow_datasets as tfds

class MultiNews(Base):
    
    def __init__(self, path):
        super().__init__(path, 'multi_news')
        return

    def dataset(self):
        return self._load('multi_news')
    
    def getGenerator(self, trainingSet, type = 'source'): 
        if type == 'source':
            return (document.numpy() for document, summary in trainingSet)

        return (summary.numpy() for document, summary in trainingSet)
