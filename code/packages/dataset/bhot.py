from .baseCsv import BaseCsv

class BHOT(BaseCsv):
    
    def __init__(self, path):
        super().__init__(path, 'bhot')
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        return

