from .base import Base
import tensorflow_datasets as tfds

class ScientificPapers(Base):
    
    def __init__(self, path, config, supervised = True):
        self.config = config
        super().__init__(path, config, supervised)
        return

    def dataset(self):
        return self._load(self.config)
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (article.numpy() for article, abstract in trainingSet)
        
        return (abstract.numpy() for article, abstract in trainingSet)
    
