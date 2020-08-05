from .base import Base
import tensorflow_datasets as tfds

class PtToEnTranslate(Base):
    
    def __init__(self, path, supervised = True):
        super().__init__(path, 'pt_to_en_translate', supervised)
        return

    def dataset(self):
        return self._load('ted_hrlr_translate/pt_to_en')
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (pt.numpy() for pt, en in trainingSet)
                
        return (en.numpy() for pt, en in trainingSet)
