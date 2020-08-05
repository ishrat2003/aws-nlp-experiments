from .baseCsv import BaseCsv

class PharmaNews(BaseCsv):
    
    def __init__(self, path):
        super().__init__(path, 'pharma_news')
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        return

