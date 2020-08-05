from .base import Base
import tensorflow_datasets as tfds

class Newsroom(Base):
    
    def __init__(self, path, supervised = True):
        super().__init__(path, 'newsroom', supervised)
        return

    def dataset(self):
        return self._load('newsroom')
    
    def getGenerator(self, trainingSet, type = 'source'):
        if (not self.supervised):
            return (item for item in trainingSet)
        
        if type == 'source':
            return (self.processSource(text.numpy()) for text, summary in trainingSet)
        
        return (self.processTarget(summary.numpy()) for text, summary in trainingSet)
    
