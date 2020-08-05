from .base import Base
import tensorflow_datasets as tfds

class WikiHow(Base):
    
    def __init__(self, path):
        super().__init__(path, 'wikihow')
        return

    def dataset(self):
        return self._load('wikihow')
    
    def getGenerator(self, trainingSet, type = 'source'): 
        if type == 'source':
            return (text.numpy() for headline, text, title in trainingSet)

        return (headline.numpy() for headline, text, title in trainingSet)
