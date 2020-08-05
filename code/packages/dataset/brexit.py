from .baseCsv import BaseCsv

class Brexit(BaseCsv):
    
    def __init__(self, path):
        super().__init__(path, 'brexit')
        return

